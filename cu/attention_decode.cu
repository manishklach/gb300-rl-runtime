#include "attention_decode.h"
#include "prefetch.h"
#include <math.h>

__device__ static float
decode_synth_query_component(const Descriptor *desc,
                             const SampleState *sample_st,
                             uint32_t idx)
{
    uint64_t seed = desc->seq_id ^ desc->reward_cookie ^
                    ((uint64_t)(idx + 1U) * 0x9e3779b97f4a7c15ULL);
    if (sample_st)
        seed ^= sample_st->rng_state[idx & 3U];
    uint32_t bits = (uint32_t)(seed ^ (seed >> 32));
    int centered = (int)(bits & 0xFFU) - 127;
    return (float)centered / 127.0f;
}

__device__ static void
decode_load_query(float *q_vec,
                  const Descriptor *desc,
                  const DecodeStepArgs *args,
                  const SampleState *sample_st)
{
    if (args && args->q_ptr) {
        const __half *q_half = (const __half *)args->q_ptr;
        for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++)
            q_vec[d] = __half2float(q_half[d]);
        return;
    }

    for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++)
        q_vec[d] = decode_synth_query_component(desc, sample_st, d);
}

extern "C" DecodeMicrokernelConfig
attention_decode_config_fixed128(void)
{
    DecodeMicrokernelConfig cfg;
    cfg.head_dim = DECODE_FIXED_HEAD_DIM;
    cfg.tokens_per_block = KV_LAYOUT_TOKENS_PER_BLOCK;
    cfg.tile_tokens = DECODE_TILE_TOKENS;
    cfg.prefetch_stages = DECODE_PREFETCH_STAGES;
    cfg.shared_bytes = DECODE_PREFETCH_STAGES * KV_LAYOUT_BLOCK_BYTES;
    return cfg;
}

__device__ DecodeStepResult
attention_decode_step_fixed128(const Descriptor *desc,
                               const DecodeStepArgs *args,
                               const uint8_t *kv_block_base,
                               SampleState *sample_st,
                               uint8_t *smem_buf)
{
    float q_vec[DECODE_FIXED_HEAD_DIM];
    float out_vec[DECODE_FIXED_HEAD_DIM];
    float scores[KV_LAYOUT_TOKENS_PER_BLOCK];
    float probs[KV_LAYOUT_TOKENS_PER_BLOCK];

    decode_load_query(q_vec, desc, args, sample_st);
    prefetch_issue(kv_block_base, smem_buf);
    prefetch_wait();

    uint32_t seq_len = (args && args->seq_len != 0U) ? args->seq_len : 1U;
    if (seq_len > KV_LAYOUT_TOKENS_PER_BLOCK)
        seq_len = KV_LAYOUT_TOKENS_PER_BLOCK;

    const __half *k_block = (const __half *)smem_buf;
    const __half *v_block = (const __half *)(smem_buf + kv_layout_v_plane_base_bytes());
    const float scale = rsqrtf((float)DECODE_FIXED_HEAD_DIM);

    float max_score = -CUDART_INF_F;
    for (uint32_t t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        const __half *k_row = k_block + t * DECODE_FIXED_HEAD_DIM;
        for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++)
            dot += q_vec[d] * __half2float(k_row[d]);
        scores[t] = dot * scale;
        if (scores[t] > max_score)
            max_score = scores[t];
    }

    float denom = 0.0f;
    for (uint32_t t = 0; t < seq_len; t++) {
        probs[t] = expf(scores[t] - max_score);
        denom += probs[t];
    }
    if (denom == 0.0f)
        denom = 1.0f;

    for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++)
        out_vec[d] = 0.0f;

    for (uint32_t t = 0; t < seq_len; t++) {
        const float p = probs[t] / denom;
        const __half *v_row = v_block + t * DECODE_FIXED_HEAD_DIM;
        for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++)
            out_vec[d] += p * __half2float(v_row[d]);
    }

    if (args && args->o_ptr) {
        float *out_ptr = (float *)args->o_ptr;
        for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++)
            out_ptr[d] = out_vec[d];
    }

    uint32_t token_id = 0;
    float best = out_vec[0];
    for (uint32_t d = 1; d < DECODE_FIXED_HEAD_DIM; d++) {
        if (out_vec[d] > best) {
            best = out_vec[d];
            token_id = d;
        }
    }

    DecodeStepResult result;
    result.path_kind = DECODE_PATH_FIXED128;
    result.bytes_touched = seq_len * DECODE_FIXED_HEAD_DIM * KV_LAYOUT_SCALAR_BYTES * 2U;
    result.tile_count = (seq_len + DECODE_TILE_TOKENS - 1U) / DECODE_TILE_TOKENS;
    result.cycle_estimate = (uint64_t)seq_len * DECODE_FIXED_HEAD_DIM * 4ULL;
    result.token_id = token_id;
    return result;
}
