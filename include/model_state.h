#pragma once

#include "attention_decode.h"
#include <cuda_fp16.h>
#include <stdint.h>

/*
 * v0.2.2-d model-state scaffold.
 *
 * This module owns the explicit hidden-state preparation/update stage
 * that sits ahead of query projection.  It is still synthetic, but the
 * runtime boundary is now:
 *
 *   descriptor/step metadata -> model_state -> query_producer -> decode
 *
 * That is closer to a real inference pipeline than directly
 * manufacturing q vectors inside the projection stage.
 */

#define MODEL_STATE_DIM DECODE_FIXED_HEAD_DIM

typedef struct {
    float  *hidden_buf;
    __half *input_proj;
    __half *residual_proj;
    float  *bias;
} ModelStateBuffers;

#ifdef __cplusplus
extern "C" {
#endif

int model_state_init(ModelStateBuffers *state, uint32_t slots);

void model_state_destroy(ModelStateBuffers *state);

int model_state_prepare_slot(ModelStateBuffers *state,
                             uint32_t slots,
                             uint64_t seq_id,
                             uint32_t step,
                             uint32_t slot);

#ifdef __cplusplus
}
#endif
