#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct {
    _Alignas(64) uint64_t descriptors_posted;
    _Alignas(64) uint64_t descriptors_consumed;
    _Alignas(64) uint64_t reward_handoffs;
    _Alignas(64) uint64_t ring_full_spins;
    _Alignas(64) uint64_t kv_blocks_allocated;
    _Alignas(64) uint64_t kv_blocks_freed;
    _Alignas(64) uint64_t gpu_idle_loops;
    _Alignas(64) uint64_t hotpath_mallocs_caught;
    _Alignas(64) uint64_t rollouts_completed;
    _Alignas(64) uint64_t pipeline_overflow;
} RuntimeMetrics;

#define METRIC_INC(m, x) __atomic_add_fetch(&(m)->x, 1, __ATOMIC_RELAXED)
#define METRIC_READ(m, x) __atomic_load_n(&(m)->x, __ATOMIC_RELAXED)

void  metrics_init(RuntimeMetrics *m);
void  metrics_fprintf(FILE *f, RuntimeMetrics *m, uint64_t wall_ns,
                      uint64_t n_tokens, uint64_t n_rollouts);
