/* ─── Lab 1: False Sharing ───────────────────────────────────────
 *
 * Two threads increment a counter N times each.
 *
 * Version A (SHARED):  counters share a cache line  → false sharing
 * Version B (PADDED):  counters are 64 bytes apart  → no false sharing
 *
 * Build:  gcc -O3 -lpthread bench.c -o bench
 * Usage:  ./bench [iterations_per_thread]
 *
 * You will see Version A run 5-20x slower due to MESI protocol
 * invalidation bouncing the cache line between cores.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>

/* ─── Version A: counters on the same cache line ──────────────── */

typedef struct {
  volatile uint64_t a;
  volatile uint64_t b;
} shared_counters;          /* likely on one cache line */

/* ─── Version B: counters padded to separate cache lines ──────── */

typedef struct {
  volatile uint64_t a;
  char              pad[56];      /* push 'b' to next cache line */
  volatile uint64_t b;
} padded_counters;          /* each field on its own 64-byte line */

/* ─── Thread args ─────────────────────────────────────────────── */

typedef struct {
  volatile uint64_t *counter;
  uint64_t           iterations;
} thread_arg;

static void *
inc_thread(void *arg) {
  thread_arg *ta = (thread_arg *)arg;
  for (uint64_t i = 0; i < ta->iterations; i++)
    (*ta->counter)++;
  return NULL;
}

/* ─── Benchmark runner ────────────────────────────────────────── */

static double
run_bench(volatile uint64_t *cnt_a, volatile uint64_t *cnt_b,
          uint64_t n, int use_padding) {
  pthread_t t1, t2;
  thread_arg a1 = { cnt_a, n };
  thread_arg a2 = { cnt_b, n };

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  pthread_create(&t1, NULL, inc_thread, &a1);
  pthread_create(&t2, NULL, inc_thread, &a2);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
              (double)(t1.tv_nsec - t0.tv_nsec);
  return ns;
}

int
main(int argc, char **argv) {
  uint64_t n = (argc > 1) ? (uint64_t)atol(argv[1]) : 100000000ULL;

  printf("False Sharing Lab\n");
  printf("  iterations per thread:  %lu\n\n", n);

  /* ── Version A: shared cache line ── */
  shared_counters shared = {0, 0};
  double ns_shared = run_bench(&shared.a, &shared.b, n, 0);
  double ns_per_op_shared = ns_shared / (2.0 * n);

  printf("  [SHARED]  same cache line:   %.0f ms  (%.1f ns/op)\n",
         ns_shared / 1e6, ns_per_op_shared);

  /* ── Version B: padded cache lines ── */
  padded_counters padded = {0, 0};
  double ns_padded = run_bench(&padded.a, &padded.b, n, 1);
  double ns_per_op_padded = ns_padded / (2.0 * n);

  printf("  [PADDED]  separate lines:    %.0f ms  (%.1f ns/op)\n\n",
         ns_padded / 1e6, ns_per_op_padded);

  printf("  speedup:  %.1fx\n", ns_shared / ns_padded);
  printf("\n");
  printf("  What happened:\n");
  printf("    SHARED: each increment invalidates the other core's\n");
  printf("            cache line → MESI bouncing → ~1 µs per round-trip.\n");
  printf("    PADDED: each core owns its cache line → local L1 write.\n");
  printf("\n");
  printf("  This is why ring.h pads head/tail to 64 bytes — the\n");
  printf("  producer and consumer never fight over a cache line.\n");

  return 0;
}
