#pragma once

#include "descriptor.h"
#include "kv_layout.h"
#include "sample.h"
#include <stdint.h>
#include <cuda_fp16.h>

/*
 * v0.2.2-b fixed-shape decode path.
 *
 * This path performs real single-token attention math for one fixed
 * configuration:
 *   - head_dim = 128
 *   - fp16/bf16-sized KV lanes
 *   - one decode query against one staged KV block
 *
 * The runtime worker may still provide a synthesized query when no
 * explicit query buffer is attached, but the decode routine itself now
 * runs real QK / softmax / V accumulation math.
 */

#define DECODE_FIXED_HEAD_DIM      KV_LAYOUT_HEAD_DIM
#define DECODE_TILE_TOKENS         16U
#define DECODE_PREFETCH_STAGES     3U
#define DECODE_WARPS_PER_BLOCK     1U

typedef enum {
    DECODE_PATH_STUB = 0,
    DECODE_PATH_FIXED128 = 1
} DecodePathKind;

typedef struct {
    const void   *q_ptr;
    void         *o_ptr;
    uint32_t      seq_len;
    uint32_t      head_dim;
    uint32_t      kv_block_base_idx;
    uint32_t      kv_block_count;
    uint32_t      output_token_offset;
} DecodeStepArgs;

typedef struct {
    uint32_t path_kind;
    uint32_t token_id;
    uint32_t bytes_touched;
    uint32_t tile_count;
    uint64_t cycle_estimate;
} DecodeStepResult;

typedef struct {
    uint32_t head_dim;
    uint32_t tokens_per_block;
    uint32_t tile_tokens;
    uint32_t prefetch_stages;
    uint32_t shared_bytes;
} DecodeMicrokernelConfig;

#ifdef __cplusplus
extern "C" {
#endif

DecodeMicrokernelConfig attention_decode_config_fixed128(void);

#ifdef __cplusplus
}
#endif

#if defined(__CUDACC__)
__device__ DecodeStepResult
attention_decode_step_fixed128(const Descriptor *desc,
                               const DecodeStepArgs *args,
                               const uint8_t *kv_block_base,
                               SampleState *sample_st,
                               uint8_t *smem_buf);
#endif
