#include "kv_prefix.h"
#include "rollout.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    int n_branches = 10000;
    if (argc > 1) n_branches = atoi(argv[1]);

    printf("GB300 RL Runtime — COW Prefix KV Benchmark\n");
    printf("  Branches:         %d\n\n", n_branches);

    KVPrefixTable tbl;
    kv_prefix_table_init(&tbl);

    uint32_t prefix_id;
    int ret = kv_prefix_register(&tbl, 0x1000, 512, 4, &prefix_id);
    if (ret != 0) {
        fprintf(stderr, "kv_prefix_register failed\n");
        return 1;
    }
    printf("  Registered prefix: id=%u  blocks=4  tokens=512\n", prefix_id);

    size_t prefix_size = 4 * 16384;
    size_t delta_size  = 2 * 16384;

    uint64_t t0 = now_ns();
    uint32_t *branch_ids = (uint32_t *)calloc(n_branches, sizeof(uint32_t));
    if (!branch_ids) { fprintf(stderr, "malloc failed\n"); return 1; }

    for (int i = 0; i < n_branches; i++) {
        int64_t delta_offset = 0x2000 + i * 2;
        ret = kv_branch_alloc(&tbl, i, prefix_id, delta_offset, 32, 2, &branch_ids[i]);
        if (ret != 0) {
            fprintf(stderr, "kv_branch_alloc failed at %d\n", i);
            break;
        }
    }
    uint64_t t1 = now_ns();

    uint64_t alloc_ns = t1 - t0;
    size_t total_shared = prefix_size;
    size_t total_delta = n_branches * delta_size;
    size_t total_cow   = total_shared + total_delta;
    size_t total_full  = n_branches * (prefix_size + delta_size);

    printf("\n── Memory Comparison ──\n");
    printf("  Shared prefix:        %zu bytes  (%zu KB)\n",
           total_shared, total_shared / 1024);
    printf("  Per-rollout delta:    %zu bytes each\n", delta_size);
    printf("  Total COW:            %zu bytes  (%zu KB)\n",
           total_cow, total_cow / 1024);
    printf("  Total full-duplicate: %zu bytes  (%zu KB)\n",
           total_full, total_full / 1024);
    printf("  Memory saved:         %zu bytes  (%zu KB)  (%.1f%%)\n",
           total_full - total_cow, (total_full - total_cow) / 1024,
           100.0 * (1.0 - (double)total_cow / (double)total_full));

    printf("\n── Allocation Performance ──\n");
    printf("  Branch alloc time:    %lu ns total, %lu ns/op\n",
           (unsigned long)alloc_ns, (unsigned long)(alloc_ns / n_branches));

    KVPrefix *p = kv_prefix_get(&tbl, prefix_id);
    printf("  Prefix refcount:      %u  (after %d branches)\n",
           p ? p->refcnt : 0, n_branches);

    printf("\n── Branch Resolution ──\n");
    t0 = now_ns();
    int64_t total_off = 0;
    for (int i = 0; i < n_branches && branch_ids[i] != UINT32_MAX; i++) {
        total_off += kv_branch_total_offset(&tbl, branch_ids[i]);
    }
    t1 = now_ns();
    printf("  Resolve all:          %lu ns total, %lu ns/op\n",
           (unsigned long)(t1 - t0), (unsigned long)((t1 - t0) / n_branches));
    printf("  Checksum:             %ld\n", (long)total_off);

    for (int i = 0; i < n_branches && branch_ids[i] != UINT32_MAX; i++)
        kv_branch_free(&tbl, branch_ids[i]);

    free(branch_ids);
    printf("\nDone.\n");
    return 0;
}
