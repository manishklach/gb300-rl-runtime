#pragma once
#include <stdint.h>
#include <stdatomic.h>

/* Completion token written by GPU, consumed by CPU.
 * Mirror of CommandRing with producer/consumer roles reversed. */

#define COMP_RING_SIZE 4096

#pragma pack(push, 1)
typedef struct {
  uint64_t seq_id;
  uint32_t token_id;        /* generated token (or error code) */
  uint32_t kv_block_offset; /* freed KV block (for reuse) */
  uint64_t reward_cookie;   /* matched to request */
  uint64_t cycles_taken;    /* SM clock cycles for this step */
} Completion;
#pragma pack(pop)

typedef struct __attribute__((packed)) {
  volatile uint32_t head __attribute__((aligned(64)));
  uint8_t           pad[60];
} CompHead;

typedef struct __attribute__((packed)) {
  volatile uint32_t tail __attribute__((aligned(64)));
  uint8_t           pad[60];
} CompTail;

typedef struct __attribute__((packed)) {
  volatile uint32_t value __attribute__((aligned(64)));
  uint8_t           pad[60];
} CompCounter;

typedef struct {
  CompHead      cons __attribute__((aligned(64)));  /* CPU writes head */
  CompTail      prod __attribute__((aligned(64)));  /* GPU writes tail */
  CompCounter   overflow __attribute__((aligned(64)));
  Completion    slots[COMP_RING_SIZE] __attribute__((aligned(128)));
} CompletionRing;

static inline void
comp_ring_record_overflow(volatile uint32_t *value) {
#if defined(__CUDA_ARCH__)
  atomicAdd((unsigned int *)value, 1U);
#else
  __atomic_add_fetch((uint32_t *)value, 1U, __ATOMIC_RELAXED);
#endif
}

/* GPU producer */
static inline int
comp_ring_push(CompletionRing *cr, const Completion *c) {
  uint32_t head = atomic_load_explicit(&cr->cons.head, memory_order_acquire);
  uint32_t tail = atomic_load_explicit(&cr->prod.tail, memory_order_relaxed);
  if (COMP_RING_SIZE - (tail - head) < 1) {
    comp_ring_record_overflow(&cr->overflow.value);
    return -1;
  }
  uint32_t pos = tail & (COMP_RING_SIZE - 1);
  cr->slots[pos] = *c;
  atomic_store_explicit(&cr->prod.tail, tail + 1, memory_order_release);
  return 0;
}

/* CPU consumer */
static inline int
comp_ring_poll(CompletionRing *cr, Completion *out) {
  uint32_t head = atomic_load_explicit(&cr->cons.head, memory_order_relaxed);
  uint32_t tail = atomic_load_explicit(&cr->prod.tail, memory_order_acquire);
  if (head >= tail)
    return 0;
  *out = cr->slots[head & (COMP_RING_SIZE - 1)];
  atomic_store_explicit(&cr->cons.head, head + 1, memory_order_release);
  return 1;
}
