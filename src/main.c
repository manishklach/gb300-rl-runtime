#include "ring.h"
#include "arena.h"
#include "completion.h"
#include "numa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

/* ─── Host-side runtime ──────────────────────────────────────────
 *
 * Initialises NVLink-C2C coherent command ring, KV arena, completion
 * ring, then launches persistent GPU workers.  The host loop publishes
 * descriptors and polls completions. */

typedef struct {
  CommandRing    *cmd_ring;
  CompletionRing *comp_ring;
  KVArena         kv_arena;
  SampleState    *d_sample_st;
  uint64_t       *d_step_count;

  int    num_sms;
  int    dev_id;
} Runtime;

static int
get_sm_count(int dev_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev_id);
  return prop.multiProcessorCount;
}

int
runtime_init(Runtime *rt, int dev_id, size_t arena_size, size_t block_size) {
  rt->dev_id = dev_id;
  cudaSetDevice(dev_id);
  rt->num_sms = get_sm_count(dev_id);

  /* allocate command ring in NUMA-local coherent memory */
  rt->cmd_ring = ring_create();
  if (!rt->cmd_ring) {
    fprintf(stderr, "failed to create command ring\n");
    return -1;
  }

  /* allocate completion ring (also coherent) */
  rt->comp_ring = (CompletionRing *)ring_create();
  if (!rt->comp_ring) {
    fprintf(stderr, "failed to create completion ring\n");
    return -1;
  }

  /* initialise KV arena */
  arena_init(&rt->kv_arena, arena_size, block_size);

  /* allocate device-side step counter */
  cudaMalloc(&rt->d_step_count, sizeof(uint64_t));
  cudaMemset(rt->d_step_count, 0, sizeof(uint64_t));

  /* allocate device-side sampling state (one per trajectory, here 1) */
  cudaMalloc(&rt->d_sample_st, sizeof(SampleState));
  SampleState h_st;
  memset(&h_st, 0, sizeof(h_st));
  h_st.rng_state[0] = 42;
  h_st.rng_state[1] = 42 ^ 0x9e3779b97f4a7c15ULL;
  h_st.rng_state[2] = (42 << 17) ^ 0x3c6ef372fe94f82aULL;
  h_st.rng_state[3] = ~42ULL;
  h_st.temperature   = 1.0f;
  h_st.top_k         = 50;
  h_st.top_p         = 0.9f;
  h_st.vocab_size    = MAX_VOCAB_SIZE;
  cudaMemcpy(rt->d_sample_st, &h_st, sizeof(SampleState),
             cudaMemcpyHostToDevice);

  return 0;
}

/* Dispatch one trajectory of n_steps decode steps through the ring. */
int
runtime_dispatch(Runtime *rt, uint64_t seq_id, int n_steps, int kv_blocks) {
  for (int i = 0; i < n_steps; i++) {
    uint32_t pos = ring_acquire(rt->cmd_ring, 1);
    if (pos == UINT32_MAX) {
      fprintf(stderr, "ring full at step %d\n", i);
      return -1;
    }

    Descriptor desc;
    desc.seq_id            = seq_id;
    desc.kv_block_offset   = (uint32_t)(i % kv_blocks);
    desc.num_kv_blocks     = 1;
    desc.attention_flags   = 0;
    desc.pad               = 0;
    desc.output_token_offset = (uint32_t)i;
    desc.reward_cookie     = (uint64_t)seq_id << 32 | i;

    rt->cmd_ring->slots[pos] = desc;
    ring_commit(rt->cmd_ring, 1);
  }
  return 0;
}

/* Poll for completions.  Returns number consumed. */
int
runtime_poll(Runtime *rt, int max_poll) {
  int n = 0;
  Completion c;
  while (n < max_poll && comp_ring_poll(rt->comp_ring, &c)) {
    /* in production: free KV block, update scheduler, etc. */
    (void)c;
    n++;
  }
  return n;
}

void
runtime_shutdown(Runtime *rt) {
  /* send sentinel to workers */
  Descriptor sentinel;
  memset(&sentinel, 0, sizeof(sentinel));
  sentinel.seq_id = UINT64_MAX;
  for (int i = 0; i < rt->num_sms; i++) {
    uint32_t pos = ring_acquire(rt->cmd_ring, 1);
    if (pos != UINT32_MAX) {
      rt->cmd_ring->slots[pos] = sentinel;
      ring_commit(rt->cmd_ring, 1);
    }
  }

  cudaFree(rt->d_step_count);
  cudaFree(rt->d_sample_st);
  ring_destroy(rt->cmd_ring);
  ring_destroy((CommandRing *)rt->comp_ring);
  arena_destroy(&rt->kv_arena);
}

/* ─── main (example: dispatch 10K steps, measure throughput) ──── */

int
main(int argc, char **argv) {
  int    dev_id    = 0;
  size_t arena_gb  = 1;       /* 1 GB KV arena */
  int    n_steps   = 10000;   /* total decode steps */
  int    kv_blocks = 128;

  int opt;
  while ((opt = getopt(argc, argv, "d:a:s:k:")) != -1) {
    switch (opt) {
    case 'd': dev_id    = atoi(optarg); break;
    case 'a': arena_gb  = (size_t)atol(optarg); break;
    case 's': n_steps   = atoi(optarg); break;
    case 'k': kv_blocks = atoi(optarg); break;
    }
  }

  Runtime rt;
  if (runtime_init(&rt, dev_id, arena_gb << 30, 16384) != 0)
    return 1;

  /* launch persistent worker kernel */
  int sms = rt.num_sms;
  int smem_size = 3 * 16384; /* PREFETCH_DEPTH * KV_BLOCK_SIZE */
  decode_worker<<<sms, 32, smem_size>>>(
    rt.cmd_ring, &rt.kv_arena, rt.comp_ring,
    rt.d_sample_st, rt.d_step_count);

  /* dispatch work */
  uint64_t start_ns, end_ns;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  runtime_dispatch(&rt, 1, n_steps, kv_blocks);

  /* wait for all completions */
  int total_completed = 0;
  while (total_completed < n_steps) {
    total_completed += runtime_poll(&rt, 256);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&start_ns, start, stop); /* actually ms */
  float ms = start_ns; /* cudaEventElapsedTime returns ms */

  uint64_t steps;
  cudaMemcpy(&steps, rt.d_step_count, sizeof(steps), cudaMemcpyDeviceToHost);

  printf("GB300 RL Runtime — Single-GPU Benchmark\n");
  printf("  SMs:          %d\n", sms);
  printf("  Steps:        %d\n", n_steps);
  printf("  Wall time:    %.2f ms\n", ms);
  printf("  Throughput:   %.0f tokens/s\n", n_steps / (ms / 1000.0f));
  printf("  GPU steps:    %lu\n", steps);

  runtime_shutdown(&rt);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
