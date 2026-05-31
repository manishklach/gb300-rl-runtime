#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>

#if defined(__CUDACC__)
#define RUNTIME_DEVICE __device__
#else
#define RUNTIME_DEVICE
#endif

#define REQUEST_RING_SIZE 1024
#define GPU_MAX_ROLLOUTS 256

#pragma pack(push, 1)
typedef struct {
    uint64_t request_id;
    uint32_t max_tokens;
    uint32_t kv_blocks;
    float    temperature;
    uint16_t top_k;
    float    top_p;
    uint64_t rng_seed;
} RolloutRequest;

_Static_assert(sizeof(RolloutRequest) == 34, "RolloutRequest must be 34 bytes");

typedef struct {
    uint64_t request_id;
    uint32_t rollout_id;
    uint32_t tokens_generated;
    float    reward;
    uint32_t status;
} RolloutDone;
#pragma pack(pop)

_Static_assert(sizeof(RolloutDone) == 24, "RolloutDone must be 24 bytes");

typedef struct __attribute__((packed)) {
    volatile uint32_t head __attribute__((aligned(64)));
    uint8_t           pad[60];
} ReqHead;

typedef struct __attribute__((packed)) {
    volatile uint32_t tail __attribute__((aligned(64)));
    uint8_t           pad[60];
} ReqTail;

typedef struct __attribute__((packed)) {
    volatile uint32_t value __attribute__((aligned(64)));
    uint8_t           pad[60];
} ReqCounter;

typedef struct {
    ReqHead        cons __attribute__((aligned(64)));
    ReqTail        prod __attribute__((aligned(64)));
    RolloutRequest slots[REQUEST_RING_SIZE] __attribute__((aligned(128)));
} RequestRing;

typedef struct {
    ReqHead       cons __attribute__((aligned(64)));
    ReqTail       prod __attribute__((aligned(64)));
    ReqCounter    overflow __attribute__((aligned(64)));
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
    uint32_t head = atomic_load_explicit(&r->cons.head, memory_order_acquire);
    uint32_t tail = atomic_load_explicit(&r->prod.tail, memory_order_relaxed);
    if (REQUEST_RING_SIZE - (tail - head) < 1)
        return UINT32_MAX;
    return tail & (REQUEST_RING_SIZE - 1);
}

static inline void
req_ring_commit(RequestRing *r)
{
    uint32_t tail = atomic_load_explicit(&r->prod.tail, memory_order_relaxed);
    atomic_store_explicit(&r->prod.tail, tail + 1, memory_order_release);
}

static inline void
done_ring_record_overflow(volatile uint32_t *value)
{
#if defined(__CUDA_ARCH__)
    atomicAdd((unsigned int *)value, 1U);
#else
    __atomic_add_fetch((uint32_t *)value, 1U, __ATOMIC_RELAXED);
#endif
}

/* GPU consumer: try to pop a request */
RUNTIME_DEVICE static inline int
req_ring_consume(RequestRing *r, RolloutRequest *out)
{
    uint32_t head = atomic_load_explicit(&r->cons.head, memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(&r->prod.tail, memory_order_acquire);
    if (head >= tail) return 0;
    *out = r->slots[head & (REQUEST_RING_SIZE - 1)];
    atomic_store_explicit(&r->cons.head, head + 1, memory_order_release);
    return 1;
}

/* GPU producer: push a done notification */
RUNTIME_DEVICE static inline int
done_ring_push(DoneRing *r, const RolloutDone *d)
{
    uint32_t head = atomic_load_explicit(&r->cons.head, memory_order_acquire);
    uint32_t tail = atomic_load_explicit(&r->prod.tail, memory_order_relaxed);
    if (REQUEST_RING_SIZE - (tail - head) < 1) {
        done_ring_record_overflow(&r->overflow.value);
        return -1;
    }
    uint32_t pos = tail & (REQUEST_RING_SIZE - 1);
    r->slots[pos] = *d;
    atomic_store_explicit(&r->prod.tail, tail + 1, memory_order_release);
    return 0;
}

/* CPU consumer: try to pop a done notification */
static inline int
done_ring_pop(DoneRing *r, RolloutDone *out)
{
    uint32_t head = atomic_load_explicit(&r->cons.head, memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(&r->prod.tail, memory_order_acquire);
    if (head >= tail) return 0;
    *out = r->slots[head & (REQUEST_RING_SIZE - 1)];
    atomic_store_explicit(&r->cons.head, head + 1, memory_order_release);
    return 1;
}
