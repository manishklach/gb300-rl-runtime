/* ─── Lab 5: MMIO-Style Doorbell Queue ──────────────────────────
 *
 * Simulates a device queue model:
 *
 *   1. Producer writes a descriptor into a shared ring.
 *   2. Producer writes a doorbell variable (simulating an MMIO register).
 *   3. Consumer sees the doorbell, drains descriptors from the ring.
 *
 * This is the exact pattern used by:
 *   - NVMe submission queues (+ doorbell register)
 *   - NIC TX/RX descriptor rings
 *   - GPU work queues (+ GPU/NIC doorbell)
 *   - Our GB300 command ring + completion ring
 *
 * The "doorbell" here is just a memory location, but on real hardware
 * it would be an MMIO register that triggers a device-side DMA read
 * of the updated producer index.
 *
 * Build:  gcc -O3 -lpthread bench.c -o bench
 * Usage:  ./bench [ops]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <string.h>

#define RING_SIZE  1024
#define MASK       (RING_SIZE - 1)

/* ─── Descriptor (fixed-size work unit) ───────────────────────── */

typedef struct __attribute__((packed)) {
  uint64_t  command_id;
  uint64_t  payload_addr;
  uint32_t  payload_len;
  uint32_t  flags;
} __attribute__((aligned(32))) WorkDescriptor;

/* ─── Doorbell ring: producer index in "device-visible" location ─ */

typedef struct {
  _Alignas(64) volatile uint32_t doorbell;  /* producer writes after enqueue */
  char         _pad1[60];
  _Alignas(64) volatile uint32_t consumed;  /* consumer writes after dequeue */
  char         _pad2[60];
  WorkDescriptor slots[RING_SIZE] _Alignas(128);
} DoorbellRing;

/* ─── Producer side ───────────────────────────────────────────── */

static int
doorbell_enqueue(DoorbellRing *r, uint64_t id,
                 uint64_t addr, uint32_t len) {
  uint32_t prod = r->doorbell;             /* relaxed: owned by producer */
  uint32_t cons = r->consumed;

  if ((prod - cons) >= RING_SIZE)
    return -1;                              /* ring full */

  uint32_t pos = prod & MASK;
  r->slots[pos].command_id   = id;
  r->slots[pos].payload_addr = addr;
  r->slots[pos].payload_len  = len;
  r->slots[pos].flags        = 0;

  /* ── write barrier + doorbell ──
   * On real hardware this store would go to an MMIO address
   * and trigger the device to DMA the updated producer index.
   */
  __atomic_thread_fence(__ATOMIC_RELEASE);
  r->doorbell = prod + 1;                  /* ring the doorbell */

  return 0;
}

/* ─── Consumer side ───────────────────────────────────────────── */

static int
doorbell_dequeue(DoorbellRing *r, WorkDescriptor *out) {
  uint32_t cons = r->consumed;             /* relaxed: owned by consumer */
  uint32_t prod = r->doorbell;             /* acquire: see producer's writes */

  if (cons == prod)
    return -1;                              /* ring empty */

  *out = r->slots[cons & MASK];
  __atomic_thread_fence(__ATOMIC_RELEASE);
  r->consumed = cons + 1;                  /* advance consumer index */

  return 0;
}

/* ─── Benchmark ───────────────────────────────────────────────── */

typedef struct {
  DoorbellRing *ring;
  uint64_t      count;
} bench_arg;

static void *
producer(void *arg) {
  bench_arg *ba = (bench_arg *)arg;
  for (uint64_t i = 0; i < ba->count; ) {
    if (doorbell_enqueue(ba->ring, i, 0xDEADBEEF, 4096) == 0)
      i++;
  }
  return NULL;
}

static void *
consumer(void *arg) {
  bench_arg *ba = (bench_arg *)arg;
  WorkDescriptor d;
  for (uint64_t i = 0; i < ba->count; ) {
    if (doorbell_dequeue(ba->ring, &d) == 0)
      i++;
  }
  return NULL;
}

int
main(int argc, char **argv) {
  uint64_t n = (argc > 1) ? (uint64_t)atol(argv[1]) : 5000000ULL;

  DoorbellRing r;
  memset(&r, 0, sizeof(r));

  printf("Doorbell Queue Mock Lab\n");
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

  printf("  wall time:   %.0f ms\n", ns / 1e6);
  printf("  latency:     %.1f ns/op\n", ns_per_op);
  printf("  throughput:  %.0f M ops/s\n\n", 1000.0 / ns_per_op);

  printf("  The pattern:\n");
  printf("    1. Producer fills descriptor in ring slot\n");
  printf("    2. Producer writes doorbell (MMIO in real HW)\n");
  printf("    3. Device/consumer sees doorbell change\n");
  printf("    4. Device/consumer DMA-reads the descriptor\n");
  printf("    5. Device/consumer advances its own consumer index\n\n");
  printf("  This is identical to:\n");
  printf("    - NVMe SQ / CQ doorbell model\n");
  printf("    - NIC TX / RX descriptor rings\n");
  printf("    - GPU work submission queues\n");
  printf("    - Our GB300 CommandRing / CompletionRing\n");

  return 0;
}
