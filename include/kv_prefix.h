#pragma once
#include <stdint.h>
#include <stddef.h>
#include "rollout.h"

#define MAX_PREFIXES 512
#define MAX_BRANCHES (MAX_ROLLOUTS * 2)

typedef struct __attribute__((packed)) {
    uint32_t prefix_id;
    uint32_t refcnt;
    int64_t  kv_block_offset;
    uint32_t token_len;
    uint32_t kv_block_count;
    uint32_t flags;
} KVPrefix;

_Static_assert(sizeof(KVPrefix) == 24, "KVPrefix must be 24 bytes");

typedef struct __attribute__((packed)) {
    uint32_t branch_id;
    uint32_t rollout_id;
    uint32_t prefix_id;
    int64_t  delta_kv_offset;
    uint32_t delta_len;
    uint32_t delta_block_count;
    uint32_t state;
    uint32_t flags;
} KVBranch;

_Static_assert(sizeof(KVBranch) == 28, "KVBranch must be 28 bytes");

typedef struct {
    KVPrefix  prefixes[MAX_PREFIXES];
    uint64_t  prefix_bitmap[(MAX_PREFIXES + 63) / 64];
    KVBranch  branches[MAX_BRANCHES];
    uint64_t  branch_bitmap[(MAX_BRANCHES + 63) / 64];
    uint32_t  next_branch_id;
} KVPrefixTable;

void  kv_prefix_table_init(KVPrefixTable *t);
int   kv_prefix_register(KVPrefixTable *t, int64_t kv_offset,
                         uint32_t token_len, uint32_t nblocks,
                         uint32_t *out_id);
int   kv_prefix_acquire(KVPrefixTable *t, uint32_t prefix_id);
int   kv_prefix_release(KVPrefixTable *t, uint32_t prefix_id);
KVPrefix *kv_prefix_get(KVPrefixTable *t, uint32_t prefix_id);

int   kv_branch_alloc(KVPrefixTable *t, uint32_t rollout_id,
                      uint32_t prefix_id, int64_t delta_kv_offset,
                      uint32_t delta_len, uint32_t delta_blocks,
                      uint32_t *out_branch_id);
int   kv_branch_free(KVPrefixTable *t, uint32_t branch_id);
KVBranch *kv_branch_get(KVPrefixTable *t, uint32_t branch_id);

int64_t kv_branch_total_offset(KVPrefixTable *t, uint32_t branch_id);
