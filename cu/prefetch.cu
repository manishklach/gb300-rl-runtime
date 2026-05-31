#include "prefetch.h"
#include <cuda_runtime.h>

/* In a production runtime these would be templated on block size.
 * For the prototype we fix KV_BLOCK_SIZE = 16384 bytes. */

__device__ void
prefetch_issue(const uint8_t *hbm_src, uint8_t *smem_dst) {
  /* cp.async.ca: pull from HBM into SMEM with cache-line bypass.
   * The last .L1::evictAll argument hints that this data should
   * not pollute L1. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
    "cp.async.ca.shared.global.L1::evictAll [%0], [%1], %2;\n"
    :
    : "r"(smem_dst), "l"(hbm_src), "n"(KV_BLOCK_SIZE)
    : "memory");
#else
  /* fallback: plain memcpy (slower, but works on any arch) */
  for (int i = 0; i < KV_BLOCK_SIZE; i++)
    smem_dst[i] = hbm_src[i];
#endif
}

__device__ void
prefetch_wait(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
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
  prefetch_issue(hbm_src, prefetch_pipeline_stage_ptr(state, stage_idx));
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
