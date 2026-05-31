#include "attention_decode.h"
#include "prefetch.h"

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
                               const uint8_t *kv_block_base,
                               SampleState *sample_st,
                               uint8_t *smem_buf)
{
    (void)sample_st;

    /*
     * v0.2.2-a scaffold behavior:
     * - stage exactly one KV block into shared memory
     * - compute a deterministic token ID from descriptor metadata
     * - report how much data the future real kernel would touch
     *
     * This preserves the worker/prefetch shape without claiming that
     * real attention math is implemented yet.
     */
    prefetch_issue(kv_block_base, smem_buf);
    prefetch_wait();

    DecodeStepResult result;
    result.path_kind = DECODE_PATH_FIXED128;
    result.bytes_touched = KV_LAYOUT_BLOCK_BYTES;
    result.tile_count = (desc->num_kv_blocks == 0) ? 1U : (uint32_t)desc->num_kv_blocks;
    result.cycle_estimate = 1000ULL + (uint64_t)result.tile_count * 64ULL;
    result.token_id = (uint32_t)(desc->seq_id * 2654435761ULL) ^
                      desc->output_token_offset ^
                      (uint32_t)(result.bytes_touched >> 4);
    return result;
}
