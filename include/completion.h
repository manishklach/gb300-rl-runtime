#pragma once
#include <stdint.h>
#include <stdatomic.h>

/* Completion token written by GPU, consumed by CPU.
 * Mirror of CommandRing with producer/consumer roles reversed. */

#define COMP_RING_SIZE 4096

typedef struct __attribute__((packed)) {
  uint64_t seq_id;
  uint32_t token_id;        /* generated token (or error code) */
  uint32_t kv_block_offset; /* freed KV block (for reuse) */
  uint64_t reward_cookie;   /* matched to request */
  uint64_t cycles_taken;    /* SM clock cycles for this step */
} Completion;

typedef struct __attribute__((packed)) {
  volatile uint32_t head __attribute__((aligned(64)));
  uint32_t          tail;
  uint8_t           pad[56];
} CompIndex;

typedef struct {
  CompIndex     prod __attribute__((aligned(64)));  /* GPU writes */
  CompIndex     cons __attribute__((aligned(64)));  /* CPU reads */
  Completion    slots[COMP_RING_SIZE] __attribute__((aligned(128)));
} CompletionRing;

/* GPU producer */
static inline void
comp_ring_push(CompletionRing *cr, const Completion *c) {
  uint32_t h = atomic_load_explicit(&cr->prod.head, memory_order_relaxed);
  uint32_t pos = h & (COMP_RING_SIZE - 1);
  cr->slots[pos] = *c;
  atomic_store_explicit(&cr->prod.head, h + 1, memory_order_release);
}

/* CPU consumer */
static inline int
comp_ring_poll(CompletionRing *cr, Completion *out) {
  uint32_t t = atomic_load_explicit(&cr->cons.tail, memory_order_acquire);
  uint32_t h = atomic_load_explicit(&cr->prod.head, memory_order_relaxed);
  if (t >= h)
    return 0;
  *out = cr->slots[t & (COMP_RING_SIZE - 1)];
  atomic_store_explicit(&cr->cons.tail, t + 1, memory_order_release);
  return 1;
}
