#include "ring.h"
#include "arena.h"
#include "completion.h"
#include "prefetch.h"
#include "sample.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/* The process_descriptor function called by the persistent worker.
 * This is a stub that should be replaced with the actual decode
 * attention kernel (e.g., FlashAttention-3 style).
 *
 * In production this would:
 *   1. prefetch_issue KV blocks from arena into SMEM
 *   2. prefetch_wait for completion
 *   3. run tiled attention with cp.async pipeline
 *   4. apply RoPE, causal mask, etc.
 *   5. write output token to completion slot
 */
__device__ static void
process_descriptor(const Descriptor *desc, KVArena *arena,
                   CompletionRing *comp_ring, SampleState *sample_st,
                   uint8_t *smem_buf) {
  /* load first KV block (prefetch pipeline) */
  uint8_t *kv_src = arena_block_ptr(arena, desc->kv_block_offset);
  prefetch_issue(kv_src, smem_buf);
  prefetch_wait();

  /* ─── stub: simulated decode ───
   * In production: launch tiled attention using the prefetched
   * KV data in smem_buf, write a logits vector to device mem,
   * then call sample_token to produce the output.
   *
   * For this prototype we just advance the completion ring to
   * validate the data path.
   */
  Completion comp;
  comp.seq_id          = desc->seq_id;
  comp.token_id        = (uint32_t)(desc->seq_id * 2654435761ULL); /* mock */
  comp.kv_block_offset = desc->kv_block_offset;
  comp.reward_cookie   = desc->reward_cookie;
  comp.cycles_taken    = 1000;  /* placeholder */

  comp_ring_push(comp_ring, &comp);
}

/* ─── Persistent decode worker kernel ────────────────────────────
 *
 * One persistent warp per SM.  Each warp polls the command ring,
 * processes descriptors, and writes completions.  The kernel runs
 * until a sentinel descriptor with seq_id == UINT64_MAX is seen.
 *
 * Launched with:
 *   gridDim  = (num_SMs, 1, 1)
 *   blockDim = (32, 1, 1)   — one warp per block
 *   smem     = PREFETCH_DEPTH * KV_BLOCK_SIZE
 */
__global__ void
decode_worker(CommandRing   *ring,
              KVArena       *arena,
              CompletionRing *comp_ring,
              SampleState   *sample_st,
              uint64_t      *step_count) {
  extern __shared__ uint8_t smem_buf[];

  uint32_t lane = threadIdx.x & 31;

  while (true) {
    Descriptor desc;

    /* ─── poll ring (lane 0 reads, broadcasts to warp) ─── */
    int got = 0;
    if (lane == 0)
      got = ring_consume(ring, &desc);

    /* broadcast via __sync_warp (implicit in warp, explicit for safety) */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    got = __sync_warp(got);
    if (got) {
      desc.seq_id          = __sync_warp(desc.seq_id);
      desc.kv_block_offset = __sync_warp(desc.kv_block_offset);
      desc.reward_cookie   = __sync_warp(desc.reward_cookie);
    }
#endif

    if (!got) {
      /* ring empty: yield to let other SMs run */
      __nanosleep(100);
      continue;
    }

    /* ─── sentinel: shut down ─── */
    if (desc.seq_id == UINT64_MAX)
      break;

    /* ─── process ─── */
    if (lane == 0) {
      process_descriptor(&desc, arena, comp_ring, sample_st, smem_buf);
      if (step_count)
        atomicAdd(step_count, 1ULL);
    }
  }
}
