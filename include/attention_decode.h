#pragma once

#include "descriptor.h"
#include "kv_layout.h"
#include "sample.h"
#include <stdint.h>

/*
 * v0.2.2-a decode microkernel scaffold.
 *
 * The first step is intentionally narrow and honest: one fixed-path
 * interface, one shared-memory staging plan, and one stubbed device
 * implementation that is ready to be replaced by real attention math.
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
    const uint8_t *kv_block_base;
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
                               const uint8_t *kv_block_base,
                               SampleState *sample_st,
                               uint8_t *smem_buf);
#endif
