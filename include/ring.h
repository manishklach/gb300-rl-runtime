#pragma once
#include <stdint.h>
#include <stdatomic.h>
#include "descriptor.h"

/* Consumer-owned head: next descriptor to read/release. */
typedef struct __attribute__((packed)) {
  volatile uint32_t head __attribute__((aligned(64)));
  uint8_t           pad[60];
} RingHead;

/* Producer-owned tail: next descriptor slot to publish. */
typedef struct __attribute__((packed)) {
  volatile uint32_t tail __attribute__((aligned(64)));
  uint8_t           pad[60];
} RingTail;

/* Lock-free SPSC command ring living in NVLink-C2C coherent memory.
 *
 *   CPU (producer)  --ring_acquire / ring_commit-->  slots
 *   GPU (consumer)  --ring_consume----------------->  slots
 *
 * Both indices wrap through RING_SIZE; the actual array index is
 * (index & (RING_SIZE-1)) for power-of-two sizing.
 */
typedef struct {
  RingHead      cons __attribute__((aligned(64)));
  RingTail      prod __attribute__((aligned(64)));
  Descriptor    slots[RING_SIZE] __attribute__((aligned(DESCRIPTOR_ALIGN)));
} CommandRing;

/* ─── Producer (CPU) side ──────────────────────────────────────── */

/* Acquire n contiguous descriptor slots.
 * Returns the slot index (masked) on success, UINT32_MAX if full.
 * The caller must fill slots [pos, pos+n) then call ring_commit. */
static inline uint32_t
ring_acquire(CommandRing *ring, uint32_t n) {
  uint32_t head = atomic_load_explicit(&ring->cons.head, memory_order_acquire);
  uint32_t tail = atomic_load_explicit(&ring->prod.tail, memory_order_relaxed);
  if (RING_SIZE - (tail - head) < n)
    return UINT32_MAX;
  return tail & (RING_SIZE - 1);
}

/* Commit n descriptors starting at pos.
 * Releases the slots to the consumer. */
static inline void
ring_commit(CommandRing *ring, uint32_t n) {
  uint32_t tail = atomic_load_explicit(&ring->prod.tail, memory_order_relaxed);
  atomic_store_explicit(&ring->prod.tail, tail + n, memory_order_release);
}

/* ─── Consumer (GPU) side ─────────────────────────────────────── */

/* Try to consume one descriptor.
 * Returns 1 on success (desc written), 0 if ring empty.
 * A successful consume advances the consumer-owned head. */
static inline int
ring_consume(CommandRing *ring, Descriptor *desc) {
  uint32_t head = atomic_load_explicit(&ring->cons.head, memory_order_relaxed);
  uint32_t tail = atomic_load_explicit(&ring->prod.tail, memory_order_acquire);
  if (head >= tail)
    return 0;
  *desc = ring->slots[head & (RING_SIZE - 1)];
  atomic_store_explicit(&ring->cons.head, head + 1, memory_order_release);
  return 1;
}

CommandRing *ring_create(void);
void ring_destroy(CommandRing *ring);
