#pragma once

#include "hw_ring.h"
#include "mmio.h"
#include <stdint.h>

typedef struct {
    hw_ring_t *cmdq;
    hw_ring_t *doneq;
    volatile uint32_t *doorbell;
    uint32_t gpu_group_id;
    uint32_t kv_arena_id;
} infer_hw_ctx_t;

int infer_submit_decode(infer_hw_ctx_t *ctx,
                        uint32_t rollout_id,
                        uint64_t kv_offset,
                        uint64_t delta_offset,
                        uint32_t prefix_id,
                        uint32_t seq_len,
                        uint32_t max_tokens);
