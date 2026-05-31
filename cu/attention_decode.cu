#include "attention_decode.h"
#include "prefetch.h"
#include <math.h>

typedef struct {
    float q_vec[DECODE_FIXED_HEAD_DIM];
    float out_vec[DECODE_FIXED_HEAD_DIM];
    float tile_scores[DECODE_TILE_TOKENS];
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
    prefetch_issue(kv_block_base, smem_buf);
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
    float running_max = -CUDART_INF_F;
    float running_norm = 0.0f;

    for (uint32_t d = lane; d < DECODE_FIXED_HEAD_DIM; d += 32U)
        scratch.out_vec[d] = 0.0f;
    __sync_warp();

    for (uint32_t tile_base = 0; tile_base < seq_len; tile_base += DECODE_TILE_TOKENS) {
        const uint32_t tile_count =
            (seq_len - tile_base) < DECODE_TILE_TOKENS ? (seq_len - tile_base) : DECODE_TILE_TOKENS;

        for (uint32_t local_base = 0; local_base < tile_count; local_base += DECODE_QK_ROWS_PER_WARP) {
            const uint32_t local_t = local_base + group;
            const uint32_t t = tile_base + local_t;
            float partial = 0.0f;
            if (local_t < tile_count) {
                const __half *k_row = k_block + t * DECODE_FIXED_HEAD_DIM;
                for (uint32_t d = lane_in_group; d < DECODE_FIXED_HEAD_DIM; d += DECODE_QK_GROUP_SIZE)
                    partial += scratch.q_vec[d] * __half2float(k_row[d]);
            }

            for (uint32_t offset = DECODE_QK_GROUP_SIZE / 2U; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset, DECODE_QK_GROUP_SIZE);

            if (lane_in_group == 0U && local_t < tile_count)
                scratch.tile_scores[local_t] = partial * scale;
        }
        __sync_warp();

        float tile_max = running_max;
        float scale_old = 0.0f;
        float new_norm = running_norm;
        if (lane == 0) {
            tile_max = scratch.tile_scores[0];
            for (uint32_t t = 1; t < tile_count; t++)
                tile_max = fmaxf(tile_max, scratch.tile_scores[t]);

            const float new_max = fmaxf(running_max, tile_max);
            scale_old = (running_norm == 0.0f) ? 0.0f : expf(running_max - new_max);

            float tile_norm = 0.0f;
            for (uint32_t t = 0; t < tile_count; t++)
                tile_norm += expf(scratch.tile_scores[t] - new_max);

            scratch.best_vals[0] = new_max;
            scratch.best_vals[1] = scale_old;
            scratch.best_vals[2] = running_norm * scale_old + tile_norm;
        }
        __sync_warp();

        const float new_max = __shfl_sync(0xFFFFFFFFU, scratch.best_vals[0], 0);
        scale_old = __shfl_sync(0xFFFFFFFFU, scratch.best_vals[1], 0);
        new_norm = __shfl_sync(0xFFFFFFFFU, scratch.best_vals[2], 0);

        for (uint32_t d = lane; d < DECODE_FIXED_HEAD_DIM; d += 32U) {
            float acc = scratch.out_vec[d] * scale_old;
            for (uint32_t local_t = 0; local_t < tile_count; local_t++) {
                const uint32_t t = tile_base + local_t;
                const float weight = expf(scratch.tile_scores[local_t] - new_max);
                const __half *v_row = v_block + t * DECODE_FIXED_HEAD_DIM;
                acc += weight * __half2float(v_row[d]);
            }
            scratch.out_vec[d] = acc;
        }
        __sync_warp();

        running_max = new_max;
        running_norm = new_norm;
    }

    if (running_norm == 0.0f)
        running_norm = 1.0f;

    for (uint32_t d = lane; d < DECODE_FIXED_HEAD_DIM; d += 32U) {
        scratch.out_vec[d] /= running_norm;
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
    result.cycle_estimate = (uint64_t)result.tile_count *
                            (uint64_t)DECODE_FIXED_HEAD_DIM * 24ULL;
    result.token_id = token_id;
    return result;
}
