#include "attention_decode.h"
#include "prefetch.h"
#include <math.h>

typedef struct {
    float q_vec[DECODE_FIXED_HEAD_DIM];
    float out_vec[DECODE_FIXED_HEAD_DIM];
    float scores[KV_LAYOUT_TOKENS_PER_BLOCK];
    float probs[KV_LAYOUT_TOKENS_PER_BLOCK];
    float best_vals[32];
    uint32_t best_idx[32];
} DecodeWarpScratch;

#define DECODE_QK_GROUP_SIZE 8U
#define DECODE_QK_ROWS_PER_WARP (32U / DECODE_QK_GROUP_SIZE)

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
    const uint32_t lane = threadIdx.x & 31U;
    if (args && args->q_ptr) {
        const __half *q_half = (const __half *)args->q_ptr;
        for (uint32_t d = lane; d < DECODE_FIXED_HEAD_DIM; d += 32U)
            q_vec[d] = __half2float(q_half[d]);
        return;
    }

    for (uint32_t d = lane; d < DECODE_FIXED_HEAD_DIM; d += 32U)
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
    const uint32_t lane = threadIdx.x & 31U;
    __shared__ DecodeWarpScratch scratch;

    decode_load_query(scratch.q_vec, desc, args, sample_st);
    __sync_warp();
    if (lane == 0)
        prefetch_issue(kv_block_base, smem_buf);
    __sync_warp();
    if (lane == 0)
        prefetch_wait();
    __sync_warp();

    uint32_t seq_len = (args && args->seq_len != 0U) ? args->seq_len : 1U;
    if (seq_len > KV_LAYOUT_TOKENS_PER_BLOCK)
        seq_len = KV_LAYOUT_TOKENS_PER_BLOCK;

    const __half *k_block = (const __half *)smem_buf;
    const __half *v_block = (const __half *)(smem_buf + kv_layout_v_plane_base_bytes());
    const float scale = rsqrtf((float)DECODE_FIXED_HEAD_DIM);

    const uint32_t group = lane / DECODE_QK_GROUP_SIZE;
    const uint32_t lane_in_group = lane % DECODE_QK_GROUP_SIZE;
    for (uint32_t base_t = 0; base_t < seq_len; base_t += DECODE_QK_ROWS_PER_WARP) {
        const uint32_t t = base_t + group;
        float partial = 0.0f;
        if (t < seq_len) {
            const __half *k_row = k_block + t * DECODE_FIXED_HEAD_DIM;
            for (uint32_t d = lane_in_group; d < DECODE_FIXED_HEAD_DIM; d += DECODE_QK_GROUP_SIZE)
                partial += scratch.q_vec[d] * __half2float(k_row[d]);
        }

        for (uint32_t offset = DECODE_QK_GROUP_SIZE / 2U; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset, DECODE_QK_GROUP_SIZE);

        if (lane_in_group == 0U && t < seq_len)
            scratch.scores[t] = partial * scale;
    }
    __sync_warp();

    float lane_score = lane < seq_len ? scratch.scores[lane] : -CUDART_INF_F;
    float max_score = lane_score;
    for (int offset = 16; offset > 0; offset >>= 1)
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFFU, max_score, offset));
    max_score = __shfl_sync(0xFFFFFFFFU, max_score, 0);

    float lane_prob = 0.0f;
    if (lane < seq_len) {
        lane_prob = expf(scratch.scores[lane] - max_score);
        scratch.probs[lane] = lane_prob;
    }
    __sync_warp();

    float denom = lane_prob;
    for (int offset = 16; offset > 0; offset >>= 1)
        denom += __shfl_down_sync(0xFFFFFFFFU, denom, offset);
    denom = __shfl_sync(0xFFFFFFFFU, denom, 0);
    if (denom == 0.0f)
        denom = 1.0f;

    for (uint32_t d = lane; d < DECODE_FIXED_HEAD_DIM; d += 32U) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < seq_len; t++) {
            const float p = scratch.probs[t] / denom;
            const __half *v_row = v_block + t * DECODE_FIXED_HEAD_DIM;
            acc += p * __half2float(v_row[d]);
        }
        scratch.out_vec[d] = acc;
    }
    __sync_warp();

    if (args && args->o_ptr) {
        float *out_ptr = (float *)args->o_ptr;
        for (uint32_t d = lane; d < DECODE_FIXED_HEAD_DIM; d += 32U)
            out_ptr[d] = scratch.out_vec[d];
    }
    __sync_warp();

    float lane_best = -CUDART_INF_F;
    uint32_t lane_best_idx = 0;
    for (uint32_t d = lane; d < DECODE_FIXED_HEAD_DIM; d += 32U) {
        if (scratch.out_vec[d] > lane_best) {
            lane_best = scratch.out_vec[d];
            lane_best_idx = d;
        }
    }
    scratch.best_vals[lane] = lane_best;
    scratch.best_idx[lane] = lane_best_idx;
    __sync_warp();

    uint32_t token_id = 0;
    if (lane == 0) {
        float best = scratch.best_vals[0];
        token_id = scratch.best_idx[0];
        for (uint32_t i = 1; i < 32U; i++) {
            if (scratch.best_vals[i] > best) {
                best = scratch.best_vals[i];
                token_id = scratch.best_idx[i];
            }
        }
    }
    token_id = __shfl_sync(0xFFFFFFFFU, token_id, 0);

    DecodeStepResult result;
    result.path_kind = DECODE_PATH_FIXED128;
    result.bytes_touched = seq_len * DECODE_FIXED_HEAD_DIM * KV_LAYOUT_SCALAR_BYTES * 2U;
    result.tile_count = (seq_len + DECODE_TILE_TOKENS - 1U) / DECODE_TILE_TOKENS;
    result.cycle_estimate = (uint64_t)seq_len * DECODE_FIXED_HEAD_DIM * 2ULL;
    result.token_id = token_id;
    return result;
}
