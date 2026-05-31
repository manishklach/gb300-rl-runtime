#pragma once

#include "attention_decode.h"
#include <cuda_fp16.h>
#include <stdint.h>

/*
 * v0.2.2-c query producer scaffold.
 *
 * This module projects a prepared hidden-state row into the fixed128
 * decode query buffer.  The hidden-state preparation/update stage lives
 * in `model_state.*`, so this layer is just:
 *   hidden state -> fixed projection -> query buffer
 *
 * The result is still not a full model, but the runtime now has a
 * cleaner separation between state preparation and query projection.
 */

#define QUERY_MODEL_DIM DECODE_FIXED_HEAD_DIM

#ifdef __cplusplus
extern "C" {
#endif

int query_producer_init(__half **d_proj_buf);

void query_producer_destroy(__half *d_proj_buf);

int query_producer_prepare_slot(const float *d_hidden_buf,
                                __half *d_query_buf,
                                const __half *d_proj_buf,
                                uint32_t slots,
                                uint32_t slot);

#ifdef __cplusplus
}
#endif
