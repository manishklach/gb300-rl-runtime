#include "prefetch.h"
#include "cp_async.cuh"
#include <cuda_runtime.h>

/* In a production runtime these would be templated on block size.
 * For the prototype we fix KV_BLOCK_SIZE = 16384 bytes. */

__device__ void
prefetch_issue_partial(const uint8_t *hbm_src, uint8_t *smem_dst,
                       uint32_t nbytes) {
  const uint32_t lane = threadIdx.x & 31U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  for (uint32_t off = lane * PREFETCH_CHUNK_BYTES;
       off < nbytes;
       off += 32U * PREFETCH_CHUNK_BYTES) {
    cp_async_ca_16(smem_dst + off, hbm_src + off);
  }
#else
  for (uint32_t off = lane; off < nbytes; off += 32U)
    smem_dst[off] = hbm_src[off];
  __syncthreads();
#endif
}

__device__ void
prefetch_issue(const uint8_t *hbm_src, uint8_t *smem_dst) {
  prefetch_issue_partial(hbm_src, smem_dst, KV_BLOCK_SIZE);
}

__device__ void
prefetch_wait(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();
#else
  /* on pre-Ampere, cp.async is unsupported — nothing to wait for */
  __syncthreads();
#endif
}

__device__ void
prefetch_pipeline_init(PrefetchPipelineState *state,
                       uint8_t *smem_base,
                       uint32_t stage_count,
                       uint32_t stage_bytes) {
  state->smem_base = smem_base;
  state->stage_count = stage_count;
  state->stage_bytes = stage_bytes;
}

__device__ uint8_t *
prefetch_pipeline_stage_ptr(const PrefetchPipelineState *state,
                            uint32_t stage_idx) {
  const uint32_t bounded =
      state->stage_count == 0 ? 0U : (stage_idx % state->stage_count);
  return state->smem_base + bounded * state->stage_bytes;
}

__device__ void
prefetch_pipeline_stage(PrefetchPipelineState *state,
                        uint32_t stage_idx,
                        const uint8_t *hbm_src) {
  prefetch_issue_partial(hbm_src, prefetch_pipeline_stage_ptr(state, stage_idx),
                         state->stage_bytes);
}

__device__ void
prefetch_pipeline_wait_stage(PrefetchPipelineState *state,
                             uint32_t stage_idx) {
  (void)state;
  (void)stage_idx;
  prefetch_wait();
}

/* ─── Host-side init ───────────────────────────────────────────── */

void
prefetch_init(int smem_per_sm) {
  /* For the prototype this is a no-op: the persistent worker
   * declares extern __shared__ space and we divide it among
   * prefetch slots at kernel launch.
   *
   * In production the host would:
   *   1. Query smemPerSM from cudaDeviceGetAttribute
   *   2. Reserve prefetch buffer from the available SMEM
   *   3. Configure the cp.async max outstanding groups
   */
  (void)smem_per_sm;
}
