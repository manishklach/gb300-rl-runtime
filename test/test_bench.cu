/* ─── Test + Benchmark for GB300 RL Runtime ─────────────────────
 *
 * Compile with:
 *   make test       # quick sanity
 *   make bench      # 1M token stress test
 *
 * The test validates the core data-path without requiring GB300
 * hardware — runs on any CUDA-capable GPU with compute 8.0+. */

#include "ring.h"
#include "arena.h"
#include "completion.h"
#include "prefetch.h"
#include "sample.h"
#include "rollout.h"
#include "pipeline.h"
#include "metrics.h"
#include "hotpath_guard.h"
#include "reward.h"
#include "kv_prefix.h"
#include "request.h"
#include "warp_broadcast.cuh"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

/* ─── Declare persistent worker kernel from worker.cu ──────────── */
__global__ void decode_worker(CommandRing*, KVArena*, CompletionRing*,
                              SampleState*, uint64_t*);

static int
cuda_is_available(void) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

__global__ static void
test_warp_broadcast_kernel(uint64_t *seq_out, uint32_t *offset_out,
                           uint64_t *cookie_out) {
  uint32_t lane = threadIdx.x & 31;
  uint64_t seq = (lane == 0) ? 0x123456789ABCDEF0ULL : 0ULL;
  uint32_t off = (lane == 0) ? 77U : 0U;
  uint64_t cookie = (lane == 0) ? 0x0FEDCBA987654321ULL : 0ULL;

  __sync_warp();
  seq_out[lane] = warp_broadcast_u64(seq, 0);
  offset_out[lane] = warp_broadcast_u32(off, 0);
  cookie_out[lane] = warp_broadcast_u64(cookie, 0);
}

/* ─── Test helpers ─────────────────────────────────────────────── */

static int
test_ring_basic(void) {
  printf("  test_ring_basic ... ");

  CommandRing *ring = ring_create();
  assert(ring);

  /* CPU: acquire + write + commit */
  uint32_t pos = ring_acquire(ring, 1);
  assert(pos != UINT32_MAX);
  ring->slots[pos].seq_id = 42;
  ring->slots[pos].kv_block_offset = 7;
  ring_commit(ring, 1);

  /* GPU-simulated (same thread): consume */
  Descriptor d;
  int got = ring_consume(ring, &d);
  assert(got);
  assert(d.seq_id == 42);
  assert(d.kv_block_offset == 7);

  ring_destroy(ring);
  printf("OK\n");
  return 0;
}

static int
test_ring_full(void) {
  printf("  test_ring_full ... ");

  CommandRing *ring = ring_create();
  assert(ring);

  /* fill the ring */
  uint32_t pos;
  int count = 0;
  while ((pos = ring_acquire(ring, 1)) != UINT32_MAX) {
    ring->slots[pos].seq_id = count;
    ring_commit(ring, 1);
    count++;
  }
  assert(count > 0); /* we got some slots */
  printf("capacity = %d, ", count);

  /* drain */
  Descriptor d;
  int drained = 0;
  while (ring_consume(ring, &d))
    drained++;
  assert(drained == count);
  printf("drained = %d OK\n", drained);

  ring_destroy(ring);
  return 0;
}

static int
test_arena_basic(void) {
  printf("  test_arena_basic ... ");

  KVArena a;
  arena_init(&a, 64UL << 20, 16384);  /* 64 MB, 16 KB blocks */
  assert(a.base);

  /* acquire / release cycling */
  int64_t ids[100];
  for (int i = 0; i < 100; i++) {
    ids[i] = arena_acquire(&a);
    assert(ids[i] >= 0);
    /* write to fault the page */
    memset(arena_block_ptr(&a, ids[i]), 0xAB, 128);
  }
  for (int i = 0; i < 100; i++)
    arena_release(&a, ids[i]);

  /* re-acquire to verify reuse */
  for (int i = 0; i < 100; i++) {
    int64_t id = arena_acquire(&a);
    assert(id >= 0);
  }

  arena_destroy(&a);
  printf("OK\n");
  return 0;
}

static int
test_completion_ring(void) {
  printf("  test_completion_ring ... ");

  CompletionRing cr;
  memset(&cr, 0, sizeof(cr));

  Completion c_in = {.seq_id = 7, .token_id = 123,
                     .kv_block_offset = 4, .reward_cookie = 0xDEAD};
  assert(comp_ring_push(&cr, &c_in) == 0);

  Completion c_out;
  int got = comp_ring_poll(&cr, &c_out);
  assert(got);
  assert(c_out.seq_id == 7);
  assert(c_out.token_id == 123);
  assert(c_out.kv_block_offset == 4);
  assert(c_out.reward_cookie == 0xDEAD);

  printf("OK\n");
  return 0;
}

static int
test_completion_ring_overflow(void) {
  printf("  test_completion_ring_overflow ... ");

  CompletionRing cr;
  memset(&cr, 0, sizeof(cr));

  Completion c = {.seq_id = 7, .token_id = 123,
                  .kv_block_offset = 4, .reward_cookie = 0xDEAD};
  for (uint32_t i = 0; i < COMP_RING_SIZE; i++)
    assert(comp_ring_push(&cr, &c) == 0);

  assert(comp_ring_push(&cr, &c) == -1);
  assert(cr.overflow.value == 1);
  assert(comp_ring_poll(&cr, &c) == 1);
  assert(comp_ring_push(&cr, &c) == 0);

  printf("OK\n");
  return 0;
}

static int
test_done_ring_overflow(void) {
  printf("  test_done_ring_overflow ... ");

  DoneRing dr;
  memset(&dr, 0, sizeof(dr));

  RolloutDone done = {.request_id = 1, .rollout_id = 2,
                      .tokens_generated = 3, .reward = 0.5f, .status = 0};
  for (uint32_t i = 0; i < REQUEST_RING_SIZE; i++)
    assert(done_ring_push(&dr, &done) == 0);

  assert(done_ring_push(&dr, &done) == -1);
  assert(dr.overflow.value == 1);
  assert(done_ring_pop(&dr, &done) == 1);
  assert(done_ring_push(&dr, &done) == 0);

  printf("OK\n");
  return 0;
}

/* ─── Bench: ring throughput, no GPU ───────────────────────────── */

static void
bench_ring(void) {
  const int N = 10000000;  /* 10M iterations */
  CommandRing *ring = ring_create();
  assert(ring);

  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  for (int i = 0; i < N; i++) {
    uint32_t pos = ring_acquire(ring, 1);
    if (pos == UINT32_MAX) {
      Descriptor d;
      ring_consume(ring, &d);
      pos = ring_acquire(ring, 1);
      assert(pos != UINT32_MAX);
    }
    ring->slots[pos].seq_id = i;
    ring_commit(ring, 1);
    Descriptor d;
    ring_consume(ring, &d);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);

  double ns = (double)(ts1.tv_sec - ts0.tv_sec) * 1e9 +
              (double)(ts1.tv_nsec - ts0.tv_nsec);
  double ns_per_op = ns / N;
  printf("  Ring throughput:         %.1f ns/op  (%.0f M ops/s)\n",
         ns_per_op, 1000.0 / ns_per_op);
  ring_destroy(ring);
}

/* ─── Full pipeline GPU test ───────────────────────────────────── */

static void
bench_full_pipeline(int n_tokens) {
  printf("\n  Full pipeline (%d tokens) ...\n", n_tokens);

  int dev_id = 0;
  cudaSetDevice(dev_id);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev_id);
  int sms = prop.multiProcessorCount;

  /* allocate rings */
  CommandRing    *cmd_ring   = ring_create();
  CompletionRing *comp_ring  = (CompletionRing *)ring_create();
  assert(cmd_ring && comp_ring);

  /* KV arena: 256 MB */
  KVArena arena;
  arena_init(&arena, 256UL << 20, 16384);

  /* device state */
  uint64_t *d_step_count;
  cudaMalloc(&d_step_count, sizeof(uint64_t));
  cudaMemset(d_step_count, 0, sizeof(uint64_t));

  SampleState *d_sample_st;
  cudaMalloc(&d_sample_st, sizeof(SampleState));
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
  cudaMemcpy(d_sample_st, &h_st, sizeof(SampleState), cudaMemcpyHostToDevice);

  /* launch persistent workers */
  int smem_size = 3 * 16384;
  decode_worker<<<sms, 32, smem_size>>>(
    cmd_ring, &arena, comp_ring, d_sample_st, d_step_count);

  /* pre-acquire KV blocks */
  int64_t kv_ids[128];
  for (int i = 0; i < 128; i++)
    kv_ids[i] = arena_acquire(&arena);
  (void)kv_ids;

  /* dispatch tokens */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < n_tokens; i++) {
    uint32_t pos = ring_acquire(cmd_ring, 1);
    if (pos == UINT32_MAX) {
      /* drain some completions */
      Completion c;
      while (comp_ring_poll(comp_ring, &c));
      pos = ring_acquire(cmd_ring, 1);
      assert(pos != UINT32_MAX);
    }
    Descriptor desc;
    desc.seq_id            = 1;
    desc.kv_block_offset   = (uint32_t)(i % 128);
    desc.num_kv_blocks     = 1;
    desc.attention_flags   = 0;
    desc.pad               = 0;
    desc.output_token_offset = (uint32_t)i;
    desc.reward_cookie     = i;
    cmd_ring->slots[pos] = desc;
    ring_commit(cmd_ring, 1);
  }

  /* wait for all completions */
  int completed = 0;
  while (completed < n_tokens) {
    Completion c;
    while (comp_ring_poll(comp_ring, &c))
      completed++;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  uint64_t steps;
  cudaMemcpy(&steps, d_step_count, sizeof(steps), cudaMemcpyDeviceToHost);

  printf("  Workers:      %d SMs\n", sms);
  printf("  Wall time:    %.2f ms\n", ms);
  printf("  Throughput:   %.0f tokens/s\n", n_tokens / (ms / 1000.0f));
  printf("  GPU steps:    %lu\n", steps);

  /* shutdown */
  Descriptor sentinel;
  memset(&sentinel, 0, sizeof(sentinel));
  sentinel.seq_id = UINT64_MAX;
  for (int i = 0; i < sms; i++) {
    uint32_t pos = ring_acquire(cmd_ring, 1);
    if (pos != UINT32_MAX) {
      cmd_ring->slots[pos] = sentinel;
      ring_commit(cmd_ring, 1);
    }
  }
  cudaDeviceSynchronize();

  cudaFree(d_step_count);
  cudaFree(d_sample_st);
  ring_destroy(cmd_ring);
  ring_destroy((CommandRing *)comp_ring);
  arena_destroy(&arena);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/* ─── New component tests ──────────────────────────────────────── */

static int
test_rollout_slab(void) {
  printf("  test_rollout_slab ... ");
  rollout_slab_t slab;
  rollout_slab_init(&slab);

  uint32_t ids[10];
  for (int i = 0; i < 10; i++) {
    int ret = rollout_alloc(&slab, &ids[i]);
    assert(ret == 0);
    assert(ids[i] < MAX_ROLLOUTS);
  }
  for (int i = 0; i < 10; i++)
    rollout_free(&slab, ids[i]);

  /* verify reuse */
  uint32_t id2;
  rollout_alloc(&slab, &id2);
  assert(id2 < MAX_ROLLOUTS);
  rollout_free(&slab, id2);
  printf("OK\n");
  return 0;
}

static int
test_rollout_transitions(void) {
  printf("  test_rollout_transitions ... ");
  rollout_t r;
  memset(&r, 0, sizeof(r));
  r.state = ROLL_FREE;

  /* invalid */
  assert(rollout_transition(&r, ROLL_FREE, ROLL_DECODING) != 0);
  assert(r.state == ROLL_FREE);

  /* valid chain */
  assert(rollout_transition(&r, ROLL_FREE, ROLL_PREFILL_READY) == 0);
  assert(rollout_transition(&r, ROLL_PREFILL_READY, ROLL_DECODING) == 0);
  assert(rollout_transition(&r, ROLL_DECODING, ROLL_REWARD_PENDING) == 0);
  assert(rollout_transition(&r, ROLL_REWARD_PENDING, ROLL_TRAJECTORY_READY) == 0);
  assert(rollout_transition(&r, ROLL_TRAJECTORY_READY, ROLL_DONE) == 0);
  assert(rollout_transition(&r, ROLL_DONE, ROLL_FREE) == 0);
  assert(r.state == ROLL_FREE);
  printf("OK\n");
  return 0;
}

static int
test_pipeline_rings(void) {
  printf("  test_pipeline_rings ... ");
  RolloutPipeline p;
  pipeline_init(&p);

  uint32_t rid;
  assert(rollout_alloc(&p.slab, &rid) == 0);

  /* push through all queues */
  for (int q = Q_FREE; q <= Q_DONE; q++) {
    assert(pipeline_push(&p, (pipeline_q_t)q, rid) == 0);
    uint32_t out;
    assert(pipeline_pop(&p, (pipeline_q_t)q, &out) == 0);
    assert(out == rid);
  }

  /* empty pop */
  uint32_t dummy;
  assert(pipeline_pop(&p, Q_FREE, &dummy) != 0);

  rollout_free(&p.slab, rid);
  printf("OK\n");
  return 0;
}

static int
test_metrics(void) {
  printf("  test_metrics ... ");
  RuntimeMetrics m;
  metrics_init(&m);
  assert(METRIC_READ(&m, descriptors_posted) == 0);
  METRIC_INC(&m, descriptors_posted);
  assert(METRIC_READ(&m, descriptors_posted) == 1);
  printf("OK\n");
  return 0;
}

static int
test_hotpath_guard(void) {
  printf("  test_hotpath_guard ... ");
  HotpathGuard g;
  hp_guard_init(&g);
  assert(g.malloc_count == 0);
  HP_GUARD_MALLOC(&g, 64);
  assert(g.malloc_count == 1);
  hp_guard_activate(&g);

  /* capture stderr to test violation message */
  FILE *old = stderr;
  FILE *sink = tmpfile();
  assert(sink);
  stderr = sink;
  HP_GUARD_MALLOC(&g, 64);
  HP_GUARD_FREE(&g, NULL);
  fclose(sink);
  stderr = old;

  assert(g.malloc_count == 2);
  assert(g.free_count == 1);
  printf("OK\n");
  return 0;
}

static int
test_warp_broadcast(void) {
  printf("  test_warp_broadcast ... ");
  if (!cuda_is_available()) {
    printf("SKIP (no CUDA device)\n");
    return 0;
  }

  uint64_t *d_seq = NULL, *d_cookie = NULL;
  uint32_t *d_off = NULL;
  uint64_t h_seq[32], h_cookie[32];
  uint32_t h_off[32];

  cudaMalloc(&d_seq, sizeof(h_seq));
  cudaMalloc(&d_cookie, sizeof(h_cookie));
  cudaMalloc(&d_off, sizeof(h_off));
  test_warp_broadcast_kernel<<<1, 32>>>(d_seq, d_off, d_cookie);
  cudaDeviceSynchronize();
  cudaMemcpy(h_seq, d_seq, sizeof(h_seq), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_off, d_off, sizeof(h_off), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_cookie, d_cookie, sizeof(h_cookie), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 32; i++) {
    assert(h_seq[i] == 0x123456789ABCDEF0ULL);
    assert(h_off[i] == 77U);
    assert(h_cookie[i] == 0x0FEDCBA987654321ULL);
  }

  cudaFree(d_seq);
  cudaFree(d_cookie);
  cudaFree(d_off);
  printf("OK\n");
  return 0;
}

static int
test_reward_ring(void) {
  printf("  test_reward_ring ... ");
  RewardRing rr;
  reward_ring_init(&rr);

  RewardDesc d;
  d.rollout_id     = 1;
  d.token_start    = 0;
  d.token_count    = 128;
  d.reward_model_id = 0;
  d.reward         = 0.0f;
  d.flags          = 0;

  assert(reward_push(&rr, &d) == 0);
  RewardDesc got;
  assert(reward_pop(&rr, &got) == 0);
  assert(got.rollout_id == 1);
  assert(got.token_count == 128);

  /* empty pop */
  assert(reward_pop(&rr, &got) != 0);
  printf("OK\n");
  return 0;
}

static int
test_kv_prefix(void) {
  printf("  test_kv_prefix ... ");
  KVPrefixTable t;
  kv_prefix_table_init(&t);

  uint32_t pid;
  int ret = kv_prefix_register(&t, 0x1000, 64, 4, &pid);
  assert(ret == 0);
  assert(pid < MAX_PREFIXES);

  KVPrefix *p = kv_prefix_get(&t, pid);
  assert(p);
  assert(p->refcnt == 1);
  assert(p->token_len == 64);

  assert(kv_prefix_acquire(&t, pid) == 0);
  assert(p->refcnt == 2);
  assert(kv_prefix_release(&t, pid) == 0);
  assert(p->refcnt == 1);
  assert(kv_prefix_release(&t, pid) == 0);

  uint32_t bid;
  ret = kv_branch_alloc(&t, 0, pid, 0x2000, 16, 1, &bid);
  assert(ret == 0);
  assert(bid < MAX_BRANCHES);

  int64_t off = kv_branch_total_offset(&t, bid);
  assert(off == 0x1000);

  assert(kv_branch_free(&t, bid) == 0);
  printf("OK\n");
  return 0;
}

/* ─── main ─────────────────────────────────────────────────────── */

int
main(int argc, char **argv) {
  int bench_mode = 0;
  int n_tokens   = 100000;

  if (argc > 1 && strcmp(argv[1], "--bench") == 0) {
    bench_mode = 1;
    if (argc > 2)
      n_tokens = atoi(argv[2]);
  }

  printf("GB300 RL Runtime — Test Suite\n\n");

  if (bench_mode) {
    printf("── Benchmark mode ──\n\n");
    bench_ring();
    if (cuda_is_available()) {
      cudaSetDevice(0);
      bench_full_pipeline(n_tokens);
    } else {
      printf("  GPU pipeline benchmark: SKIP (no CUDA device)\n");
    }
    printf("\nDone.\n");
    return 0;
  }

  /* unit tests */
  printf("── Unit tests ──\n\n");
  test_ring_basic();
  test_ring_full();
  test_arena_basic();
  test_completion_ring();
  test_completion_ring_overflow();
  test_rollout_slab();
  test_rollout_transitions();
  test_pipeline_rings();
  test_metrics();
  test_hotpath_guard();
  test_reward_ring();
  test_kv_prefix();
  test_done_ring_overflow();
  test_warp_broadcast();

  if (cuda_is_available()) {
    printf("\n── Quick pipeline test (1000 tokens) ──\n\n");
    bench_full_pipeline(1000);
  } else {
    printf("\n── Quick pipeline test ──\n\n");
    printf("  SKIP (no CUDA device)\n");
  }

  printf("\nAll tests passed.\n");
  return 0;
}
