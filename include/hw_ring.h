#pragma once

#include "hw_desc.h"
#include <stddef.h>
#include <stdint.h>

#define HW_RING_SIZE (1u << 16)
#define HW_RING_MASK (HW_RING_SIZE - 1u)

typedef struct {
    _Alignas(64) volatile uint32_t head;
    char pad0[60];

    _Alignas(64) volatile uint32_t tail;
    char pad1[60];

    _Alignas(64) hw_desc_t desc[HW_RING_SIZE];
} hw_ring_t;

uint32_t hw_ring_load_acquire(volatile uint32_t *p);
void hw_ring_store_release(volatile uint32_t *p, uint32_t value);

hw_ring_t *hw_ring_create(void);
void hw_ring_reset(hw_ring_t *ring);
void hw_ring_destroy(hw_ring_t *ring);

static inline int
hw_ring_full(hw_ring_t *ring)
{
    const uint32_t head = hw_ring_load_acquire(&ring->head);
    const uint32_t tail = hw_ring_load_acquire(&ring->tail);
    return (tail - head) >= HW_RING_SIZE;
}

static inline int
hw_ring_empty(hw_ring_t *ring)
{
    const uint32_t head = hw_ring_load_acquire(&ring->head);
    const uint32_t tail = hw_ring_load_acquire(&ring->tail);
    return head == tail;
}

static inline int
hw_ring_push(hw_ring_t *ring, const hw_desc_t *desc)
{
    const uint32_t tail = hw_ring_load_acquire(&ring->tail);
    const uint32_t head = hw_ring_load_acquire(&ring->head);

    if ((tail - head) >= HW_RING_SIZE)
        return -1;

    ring->desc[tail & HW_RING_MASK] = *desc;
    hw_ring_store_release(&ring->tail, tail + 1u);
    return 0;
}

static inline int
hw_ring_pop(hw_ring_t *ring, hw_desc_t *out)
{
    const uint32_t head = hw_ring_load_acquire(&ring->head);
    const uint32_t tail = hw_ring_load_acquire(&ring->tail);

    if (head == tail)
        return -1;

    *out = ring->desc[head & HW_RING_MASK];
    hw_ring_store_release(&ring->head, head + 1u);
    return 0;
}
