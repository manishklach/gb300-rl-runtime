/* ─── Lab 2: Build an SPSC Ring From Scratch ────────────────────
 *
 * A single-producer, single-consumer lock-free ring buffer.
 * The producer and consumer each own one index — no contention.
 *
 * Build:  gcc -O3 -lpthread bench.c -o bench
 * Usage:  ./bench [ops]
 *
 * This is the same pattern used in:
 *   - NVMe submission/completion queues
 *   - NIC descriptor rings
 *   - GPU command queues
 *   - io_uring
 *   - DPDK
 *   - Our GB300 runtime (ring.h)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>

#define RING_SIZE  4096
#define MASK       (RING_SIZE - 1)

/* ─── Cache-line-padded producer/consumer indices ─────────────── */

typedef struct {
  _Alignas(64) volatile uint32_t head;  /* owned by producer */
  uint32_t     tail;                     /* read by producer */
  char         _pad1[56];
} prod_idx;

typedef struct {
  _Alignas(64) volatile uint32_t tail;  /* owned by consumer */
  uint32_t     head;                    /* read by consumer */
  char         _pad2[56];
} cons_idx;

/* ─── The ring ────────────────────────────────────────────────── */

typedef struct {
  prod_idx  p;
  cons_idx  c;
  uint64_t  slots[RING_SIZE] _Alignas(128);
} Ring;

/* ─── Producer API ────────────────────────────────────────────── */

static inline int
ring_push(Ring *r, uint64_t val) {
  uint32_t h = atomic_load_explicit(&r->p.head, memory_order_relaxed);
  uint32_t t = atomic_load_explicit(&r->c.tail, memory_order_acquire);

  if ((h - t) >= RING_SIZE)
    return -1;                     /* full */

  r->slots[h & MASK] = val;
  atomic_store_explicit(&r->p.head, h + 1, memory_order_release);
  return 0;
}

/* ─── Consumer API ────────────────────────────────────────────── */

static inline int
ring_pop(Ring *r, uint64_t *out) {
  uint32_t t = atomic_load_explicit(&r->c.tail, memory_order_relaxed);
  uint32_t h = atomic_load_explicit(&r->p.head, memory_order_acquire);

  if (t == h)
    return -1;                     /* empty */

  *out = r->slots[t & MASK];
  atomic_store_explicit(&r->c.tail, t + 1, memory_order_release);
  return 0;
}

/* ─── Benchmark ───────────────────────────────────────────────── */

typedef struct {
  Ring    *ring;
  uint64_t count;
} bench_arg;

static void *
producer(void *arg) {
  bench_arg *ba = (bench_arg *)arg;
  for (uint64_t i = 0; i < ba->count; ) {
    if (ring_push(ba->ring, i) == 0)
      i++;
  }
  return NULL;
}

static void *
consumer(void *arg) {
  bench_arg *ba = (bench_arg *)arg;
  uint64_t val;
  for (uint64_t i = 0; i < ba->count; ) {
    if (ring_pop(ba->ring, &val) == 0)
      i++;
  }
  return NULL;
}

int
main(int argc, char **argv) {
  uint64_t n = (argc > 1) ? (uint64_t)atol(argv[1]) : 50000000ULL;

  Ring r = {0};

  printf("SPSC Ring Lab\n");
  printf("  ops:  %lu\n\n", n);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  pthread_t prod, cons;
  bench_arg ba = { &r, n };
  pthread_create(&prod, NULL, producer, &ba);
  pthread_create(&cons, NULL, consumer, &ba);
  pthread_join(prod, NULL);
  pthread_join(cons, NULL);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
              (double)(t1.tv_nsec - t0.tv_nsec);
  double ns_per_op = ns / n;

  printf("  wall time:  %.0f ms\n", ns / 1e6);
  printf("  latency:    %.1f ns/op\n", ns_per_op);
  printf("  throughput: %.0f M ops/s\n", 1000.0 / ns_per_op);
  printf("\n");
  printf("  Key observations:\n");
  printf("    - No locks.  Producer and consumer each own one index.\n");
  printf("    - acquire/release atomics prevent reordering.\n");
  printf("    - Cache line padding eliminates false sharing.\n");
  printf("    - This exact pattern drives NVMe, NICs, and GPU rings.\n");

  return 0;
}
