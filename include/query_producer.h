#pragma once

#include "attention_decode.h"
#include <cuda_fp16.h>
#include <stdint.h>

/*
 * v0.2.2-c query producer scaffold.
 *
 * This module provides a tiny explicit "model-like" query production
 * stage so the runtime path no longer host-fills decode queries
 * directly.  It is intentionally narrow:
 *   - one hidden-state size
 *   - one fixed projection matrix
 *   - deterministic synthetic hidden-state initialization/update
 *
 * The result is still not a full model, but the runtime now has an
 * explicit producer stage between control metadata and decode queries.
 */

#define QUERY_MODEL_DIM DECODE_FIXED_HEAD_DIM

#ifdef __cplusplus
extern "C" {
#endif

int query_producer_init(float **d_hidden_buf,
                        __half **d_proj_buf,
                        uint32_t slots);

void query_producer_destroy(float *d_hidden_buf,
                            __half *d_proj_buf);

int query_producer_prepare_slot(float *d_hidden_buf,
                                __half *d_query_buf,
                                const __half *d_proj_buf,
                                uint32_t slots,
                                uint64_t seq_id,
                                uint32_t step,
                                uint32_t slot);

#ifdef __cplusplus
}
#endif
