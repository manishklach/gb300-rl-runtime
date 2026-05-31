#include "metrics.h"
#include <stdio.h>
#include <string.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

void
metrics_init(RuntimeMetrics *m)
{
    memset(m, 0, sizeof(*m));
}

void
metrics_fprintf(FILE *f, RuntimeMetrics *m, uint64_t wall_ns,
                uint64_t n_tokens, uint64_t n_rollouts)
{
    uint64_t posted    = METRIC_READ(m, descriptors_posted);
    uint64_t consumed  = METRIC_READ(m, descriptors_consumed);
    uint64_t handoffs  = METRIC_READ(m, reward_handoffs);
    uint64_t spins     = METRIC_READ(m, ring_full_spins);
    uint64_t comp_ovf  = METRIC_READ(m, completion_overflow_attempts);
    uint64_t done_ovf  = METRIC_READ(m, done_overflow_attempts);
    uint64_t kv_alloc  = METRIC_READ(m, kv_blocks_allocated);
    uint64_t kv_freed  = METRIC_READ(m, kv_blocks_freed);
    uint64_t idle      = METRIC_READ(m, gpu_idle_loops);
    uint64_t hp_malloc = METRIC_READ(m, hotpath_mallocs_caught);
    uint64_t done      = METRIC_READ(m, rollouts_completed);
    uint64_t overflow  = METRIC_READ(m, pipeline_overflow);

    double wall_s = wall_ns / 1.0e9;
    double tok_s  = n_tokens / wall_s;

    fprintf(f, "\n── Pipeline Metrics ──\n");
    fprintf(f, "  Wall time:              %.3f s\n", wall_s);
    fprintf(f, "  Tokens:                 %lu\n", (unsigned long)n_tokens);
    fprintf(f, "  Rollouts:               %lu\n", (unsigned long)n_rollouts);
    fprintf(f, "  Throughput:             %.0f tokens/s\n", tok_s);
    fprintf(f, "  Descriptors posted:     %lu\n", (unsigned long)posted);
    fprintf(f, "  Descriptors consumed:   %lu\n", (unsigned long)consumed);
    fprintf(f, "  Reward handoffs:        %lu\n", (unsigned long)handoffs);
    fprintf(f, "  Ring full spins:        %lu\n", (unsigned long)spins);
    fprintf(f, "  Completion overflows:   %lu\n", (unsigned long)comp_ovf);
    fprintf(f, "  Done ring overflows:    %lu\n", (unsigned long)done_ovf);
    fprintf(f, "  KV blocks allocated:    %lu\n", (unsigned long)kv_alloc);
    fprintf(f, "  KV blocks freed:        %lu\n", (unsigned long)kv_freed);
    fprintf(f, "  KV arena utilization:   %lu / %lu\n",
            (unsigned long)(kv_alloc - kv_freed), (unsigned long)kv_alloc);
    fprintf(f, "  GPU idle loops:         %lu\n", (unsigned long)idle);
    fprintf(f, "  Rollouts completed:     %lu\n", (unsigned long)done);
    fprintf(f, "  Pipeline overflows:     %lu\n", (unsigned long)overflow);
    fprintf(f, "  Wrapper-tracked mallocs:%lu ", (unsigned long)hp_malloc);
    if (hp_malloc == 0)
        fprintf(f, "(wrapper-clean)\n");
    else
        fprintf(f, "(WRAPPER VIOLATION)\n");
    fflush(f);
}

void
metrics_snapshot_page_faults(PageFaultSnapshot *snap)
{
    memset(snap, 0, sizeof(*snap));
#if defined(__linux__) || defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        snap->minor_faults = (uint64_t)usage.ru_minflt;
        snap->major_faults = (uint64_t)usage.ru_majflt;
        snap->supported = 1;
    }
#endif
}

void
metrics_diff_page_faults(PageFaultSnapshot *delta,
                         const PageFaultSnapshot *start,
                         const PageFaultSnapshot *end)
{
    memset(delta, 0, sizeof(*delta));
    if (!start->supported || !end->supported)
        return;
    delta->minor_faults = end->minor_faults - start->minor_faults;
    delta->major_faults = end->major_faults - start->major_faults;
    delta->supported = 1;
}
