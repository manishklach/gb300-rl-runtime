#include "ring.h"
#include "arena.h"
#include "completion.h"
#include "attention_decode.h"
#include "prefetch.h"
#include "sample.h"
#include "warp_broadcast.cuh"
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
  uint8_t *kv_src = arena_block_ptr(arena, desc->kv_block_offset);
  DecodeStepResult result =
      attention_decode_step_fixed128(desc, kv_src, sample_st, smem_buf);

  Completion comp;
  comp.seq_id          = desc->seq_id;
  comp.token_id        = result.token_id;
  comp.kv_block_offset = desc->kv_block_offset;
  comp.reward_cookie   = desc->reward_cookie;
  comp.cycles_taken    = result.cycle_estimate;

  while (comp_ring_push(comp_ring, &comp) != 0)
    __nanosleep(100);
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

    /* Use __shfl_sync for value broadcast and __sync_warp only as a barrier. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __sync_warp();
    got = (int)warp_broadcast_u32((uint32_t)got, 0);
    if (got) {
      desc.seq_id            = warp_broadcast_u64(desc.seq_id, 0);
      desc.kv_block_offset   = warp_broadcast_u32(desc.kv_block_offset, 0);
      desc.num_kv_blocks     = (uint16_t)warp_broadcast_u32(desc.num_kv_blocks, 0);
      desc.attention_flags   = (uint8_t)warp_broadcast_u32(desc.attention_flags, 0);
      desc.pad               = (uint8_t)warp_broadcast_u32(desc.pad, 0);
      desc.output_token_offset = warp_broadcast_u32(desc.output_token_offset, 0);
      desc.reward_cookie     = warp_broadcast_u64(desc.reward_cookie, 0);
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
