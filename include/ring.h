#pragma once
#include <stdint.h>
#include <stdatomic.h>
#include "descriptor.h"

/* Cache-line-padded producer/consumer index pair (one per direction). */
typedef struct __attribute__((packed)) {
  volatile uint32_t head __attribute__((aligned(64)));
  uint32_t          tail;
  uint8_t           pad[56];
} RingHeadTail;

/* Lock-free SPSC command ring living in NVLink-C2C coherent memory.
 *
 *   CPU (producer)  --ring_acquire / ring_commit-->  slots
 *   GPU (consumer)  --ring_consume / ring_release-->  slots
 *
 * Both indices wrap through RING_SIZE; the actual array index is
 * (index & (RING_SIZE-1)) for power-of-two sizing.
 */
typedef struct {
  RingHeadTail  hta __attribute__((aligned(64)));
  RingHeadTail  htb __attribute__((aligned(64)));
  Descriptor    slots[RING_SIZE] __attribute__((aligned(DESCRIPTOR_ALIGN)));
} CommandRing;

/* ─── Producer (CPU) side ──────────────────────────────────────── */

/* Acquire n contiguous descriptor slots.
 * Returns the slot index (masked) on success, UINT32_MAX if full.
 * The caller must fill slots [pos, pos+n) then call ring_commit. */
static inline uint32_t
ring_acquire(CommandRing *ring, uint32_t n) {
  uint32_t h = atomic_load_explicit(&ring->hta.head, memory_order_acquire);
  uint32_t t = atomic_load_explicit(&ring->hta.tail, memory_order_relaxed);
  if (RING_SIZE - (h - t) < n)
    return UINT32_MAX;
  return h & (RING_SIZE - 1);
}

/* Commit n descriptors starting at pos.
 * Releases the slots to the consumer. */
static inline void
ring_commit(CommandRing *ring, uint32_t n) {
  uint32_t h = atomic_load_explicit(&ring->hta.head, memory_order_relaxed);
  atomic_store_explicit(&ring->hta.head, h + n, memory_order_release);
}

/* ─── Consumer (GPU) side ─────────────────────────────────────── */

/* Try to consume one descriptor.
 * Returns 1 on success (desc written), 0 if ring empty.
 * Call ring_release after processing to advance the consumer index. */
static inline int
ring_consume(const CommandRing *ring, Descriptor *desc) {
  uint32_t t = atomic_load_explicit(&ring->htb.tail, memory_order_acquire);
  uint32_t h = atomic_load_explicit(&ring->hta.head, memory_order_relaxed);
  if (t >= h)
    return 0;
  *desc = ring->slots[t & (RING_SIZE - 1)];
  atomic_store_explicit(&ring->htb.tail, t + 1, memory_order_release);
  return 1;
}
