#pragma once
#include <stdint.h>
#include <stddef.h>

#define TRACE_CAPACITY (1UL << 20)

typedef enum {
    TRACE_ROLLOUT_ALLOC,
    TRACE_ROLLOUT_FREE,
    TRACE_DESC_POSTED,
    TRACE_DESC_COMMITTED,
    TRACE_DESC_CONSUMED,
    TRACE_COMPLETION_POSTED,
    TRACE_COMPLETION_POLLED,
    TRACE_REWARD_POSTED,
    TRACE_REWARD_SCORED,
    TRACE_TRAJECTORY_DONE,
    TRACE_EVENT_COUNT,
} trace_event_t;

typedef struct __attribute__((packed)) {
    uint64_t tsc;
    uint32_t type;
    uint32_t rollout_id;
    uint32_t seq;
} trace_entry_t;

_Static_assert(sizeof(trace_entry_t) == 20, "trace_entry_t must be 20 bytes");

typedef struct {
    trace_entry_t entries[TRACE_CAPACITY];
    volatile uint64_t head;
} TraceRing;

void  trace_init(TraceRing *r);
void  trace_push(TraceRing *r, trace_event_t type, uint32_t rollout_id, uint32_t seq);
void  trace_push_ts(TraceRing *r, uint64_t tsc, trace_event_t type,
                    uint32_t rollout_id, uint32_t seq);

int   trace_report_from(const TraceRing *r, uint64_t wall_ns, uint64_t n_tokens,
                        uint64_t n_rollouts, const char *prefix);
