#include "trace.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void
trace_init(TraceRing *r)
{
    memset(r, 0, sizeof(*r));
}

void
trace_push(TraceRing *r, trace_event_t type, uint32_t rollout_id, uint32_t seq)
{
    trace_push_ts(r, now_ns(), type, rollout_id, seq);
}

void
trace_push_ts(TraceRing *r, uint64_t tsc, trace_event_t type,
              uint32_t rollout_id, uint32_t seq)
{
    uint64_t h = __atomic_load_n(&r->head, __ATOMIC_RELAXED);
    uint32_t pos = (uint32_t)(h & (TRACE_CAPACITY - 1));
    r->entries[pos].tsc        = tsc;
    r->entries[pos].type       = (uint32_t)type;
    r->entries[pos].rollout_id = rollout_id;
    r->entries[pos].seq        = seq;
    __atomic_store_n(&r->head, h + 1, __ATOMIC_RELEASE);
}

static const char *
event_name(uint32_t t)
{
    switch ((trace_event_t)t) {
    case TRACE_ROLLOUT_ALLOC:      return "ROLLOUT_ALLOC";
    case TRACE_ROLLOUT_FREE:       return "ROLLOUT_FREE";
    case TRACE_DESC_POSTED:        return "DESC_POSTED";
    case TRACE_DESC_COMMITTED:     return "DESC_COMMITTED";
    case TRACE_DESC_CONSUMED:      return "DESC_CONSUMED";
    case TRACE_COMPLETION_POSTED:  return "COMPLETION_POSTED";
    case TRACE_COMPLETION_POLLED:  return "COMPLETION_POLLED";
    case TRACE_REWARD_POSTED:      return "REWARD_POSTED";
    case TRACE_REWARD_SCORED:      return "REWARD_SCORED";
    case TRACE_TRAJECTORY_DONE:    return "TRAJECTORY_DONE";
    default:                       return "?";
    }
}

static int
cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static void
report_pair(const TraceRing *r, uint64_t count,
            trace_event_t start_ev, trace_event_t end_ev,
            const char *label)
{
    uint64_t n = 0;
    uint64_t bufsz = count;
    uint64_t *lats = (uint64_t *)calloc(bufsz, sizeof(uint64_t));
    if (!lats) return;

    for (uint64_t si = 0, ei = 0; si < count && ei < count && n < bufsz; ) {
        uint32_t st = r->entries[si].type;
        uint32_t et = r->entries[ei].type;

        if (si >= count || ei >= count) break;

        if (st == (uint32_t)start_ev && et == (uint32_t)end_ev &&
            r->entries[si].rollout_id == r->entries[ei].rollout_id &&
            r->entries[si].seq == r->entries[ei].seq &&
            r->entries[ei].tsc >= r->entries[si].tsc)
        {
            lats[n++] = r->entries[ei].tsc - r->entries[si].tsc;
            si++; ei++;
        } else if (st == (uint32_t)start_ev && et != (uint32_t)end_ev) {
            ei++;
        } else if (st != (uint32_t)start_ev) {
            si++;
        } else {
            si++; ei++;
        }
    }

    if (n == 0) {
        printf("  %-40s  no samples\n", label);
        free(lats);
        return;
    }

    qsort(lats, n, sizeof(uint64_t), cmp_u64);

    uint64_t p50 = lats[n / 2];
    uint64_t p90 = lats[n * 9 / 10];
    uint64_t p99 = lats[n * 99 / 100];
    uint64_t avg = 0;
    uint64_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += lats[i];
    avg = sum / n;

    printf("  %-40s  n=%-6lu  avg=%-5lu  p50=%-5lu  p90=%-5lu  p99=%-5lu ns\n",
           label, (unsigned long)n, (unsigned long)avg,
           (unsigned long)p50, (unsigned long)p90, (unsigned long)p99);
    free(lats);
}

int
trace_report_from(const TraceRing *r, uint64_t wall_ns, uint64_t n_tokens,
                  uint64_t n_rollouts, const char *prefix)
{
    uint64_t count = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
    if (count > TRACE_CAPACITY)
        count = TRACE_CAPACITY;

    double wall_s = wall_ns / 1.0e9;
    printf("\n%sTrace Latency Report (%lu events, %.3f s wall)\n",
           prefix ? prefix : "", (unsigned long)count, wall_s);

    report_pair(r, count, TRACE_DESC_POSTED, TRACE_DESC_COMMITTED,
                "descriptor post -> commit");
    report_pair(r, count, TRACE_DESC_POSTED, TRACE_DESC_CONSUMED,
                "descriptor post -> GPU dequeue");
    report_pair(r, count, TRACE_DESC_POSTED, TRACE_COMPLETION_POLLED,
                "descriptor post -> completion polled");
    report_pair(r, count, TRACE_DESC_CONSUMED, TRACE_COMPLETION_POLLED,
                "GPU dequeue -> completion polled");
    report_pair(r, count, TRACE_ROLLOUT_ALLOC, TRACE_REWARD_POSTED,
                "rollout alloc -> reward posted");
    report_pair(r, count, TRACE_ROLLOUT_ALLOC, TRACE_TRAJECTORY_DONE,
                "rollout alloc -> trajectory done");
    report_pair(r, count, TRACE_REWARD_POSTED, TRACE_REWARD_SCORED,
                "reward posted -> scored");
    report_pair(r, count, TRACE_REWARD_POSTED, TRACE_TRAJECTORY_DONE,
                "reward posted -> trajectory done");
    fflush(stdout);
    return 0;
}
