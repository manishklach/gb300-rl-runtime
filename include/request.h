#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <cuda_runtime.h>

#define REQUEST_RING_SIZE 1024
#define GPU_MAX_ROLLOUTS 256

typedef struct __attribute__((packed)) {
    uint64_t request_id;
    uint32_t max_tokens;
    uint32_t kv_blocks;
    float    temperature;
    uint16_t top_k;
    float    top_p;
    uint64_t rng_seed;
} RolloutRequest;

_Static_assert(sizeof(RolloutRequest) == 32, "RolloutRequest must be 32 bytes");

typedef struct __attribute__((packed)) {
    uint64_t request_id;
    uint32_t rollout_id;
    uint32_t tokens_generated;
    float    reward;
    uint32_t status;
} RolloutDone;

_Static_assert(sizeof(RolloutDone) == 24, "RolloutDone must be 24 bytes");

typedef struct __attribute__((packed)) {
    volatile uint32_t head __attribute__((aligned(64)));
    uint32_t          tail;
    uint8_t           pad[56];
} ReqIndex;

typedef struct {
    ReqIndex       prod __attribute__((aligned(64)));
    ReqIndex       cons __attribute__((aligned(64)));
    RolloutRequest slots[REQUEST_RING_SIZE] __attribute__((aligned(128)));
} RequestRing;

typedef struct {
    ReqIndex      prod __attribute__((aligned(64)));
    ReqIndex      cons __attribute__((aligned(64)));
    RolloutDone   slots[REQUEST_RING_SIZE] __attribute__((aligned(128)));
} DoneRing;

typedef struct {
    uint32_t active;
    uint32_t request_id;
    uint32_t tokens_generated;
    uint32_t max_tokens;
    uint32_t kv_blocks;
    uint64_t rng_state[4];
    float    temperature;
    uint16_t top_k;
    float    top_p;
} GpuRolloutSlot;

typedef struct {
    GpuRolloutSlot slots[GPU_MAX_ROLLOUTS];
    uint64_t       bitmap[(GPU_MAX_ROLLOUTS + 63) / 64];
} GpuRolloutState;

/* CPU producer: reserve a slot for a new request */
static inline uint32_t
req_ring_acquire(RequestRing *r)
{
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_acquire);
    uint32_t t = atomic_load_explicit(&r->cons.tail, memory_order_relaxed);
    if (REQUEST_RING_SIZE - (h - t) < 1)
        return UINT32_MAX;
    return h & (REQUEST_RING_SIZE - 1);
}

static inline void
req_ring_commit(RequestRing *r)
{
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_relaxed);
    atomic_store_explicit(&r->prod.head, h + 1, memory_order_release);
}

/* GPU consumer: try to pop a request */
__device__ static inline int
req_ring_consume(RequestRing *r, RolloutRequest *out)
{
    uint32_t t = atomic_load_explicit(&r->cons.tail, memory_order_acquire);
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_relaxed);
    if (t >= h) return 0;
    *out = r->slots[t & (REQUEST_RING_SIZE - 1)];
    atomic_store_explicit(&r->cons.tail, t + 1, memory_order_release);
    return 1;
}

/* GPU producer: push a done notification */
__device__ static inline void
done_ring_push(DoneRing *r, const RolloutDone *d)
{
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_relaxed);
    uint32_t pos = h & (REQUEST_RING_SIZE - 1);
    r->slots[pos] = *d;
    atomic_store_explicit(&r->prod.head, h + 1, memory_order_release);
}

/* CPU consumer: try to pop a done notification */
static inline int
done_ring_pop(DoneRing *r, RolloutDone *out)
{
    uint32_t t = atomic_load_explicit(&r->cons.tail, memory_order_acquire);
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_relaxed);
    if (t >= h) return 0;
    *out = r->slots[t & (REQUEST_RING_SIZE - 1)];
    atomic_store_explicit(&r->cons.tail, t + 1, memory_order_release);
    return 1;
}
