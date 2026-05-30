#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

/* Software-pipelined async copy engine for KV block prefetch.
 *
 * Uses cp.async to load KV blocks from HBM into SMEM/shared memory
 * while the previous block is being consumed by the attention kernel.
 * This decouples HBM latency from arithmetic. */

#define PREFETCH_DEPTH   3    /* triple buffering */
#define KV_BLOCK_SIZE    16384 /* bytes; must match arena block_size */

/* Prefetch pipeline slot */
typedef struct {
  char    *smem_base;        /* target in shared memory */
  uint32_t block_global_idx; /* HBM block index */
  uint32_t phase;            /* pipeline phase counter */
} PrefetchSlot;

/* Host-side: initialize the pipeline state.
 * Allocates PREFETCH_DEPTH * KV_BLOCK_SIZE of shared memory per SM. */
void prefetch_init(int smem_per_sm);

/* Device-side: issue cp.async for one KV block.
 * Called by the persistent worker before attention. */
__device__ void prefetch_issue(const uint8_t *hbm_src, uint8_t *smem_dst);

/* Device-side: wait for the oldest in-flight prefetch to complete. */
__device__ void prefetch_wait(void);
