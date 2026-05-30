#include "kv_prefix.h"
#include "rollout.h"
#include <string.h>

void
kv_prefix_table_init(KVPrefixTable *t)
{
    memset(t, 0, sizeof(*t));
    t->prefix_bitmap[0] = ~0ULL;
    for (int w = 1; w < (int)(sizeof(t->prefix_bitmap) / sizeof(t->prefix_bitmap[0])); w++)
        t->prefix_bitmap[w] = ~0ULL;
    if (MAX_PREFIXES % 64)
        t->prefix_bitmap[MAX_PREFIXES / 64] &= (1ULL << (MAX_PREFIXES % 64)) - 1;
    t->branch_bitmap[0] = ~0ULL;
    for (int w = 1; w < (int)(sizeof(t->branch_bitmap) / sizeof(t->branch_bitmap[0])); w++)
        t->branch_bitmap[w] = ~0ULL;
    if (MAX_BRANCHES % 64)
        t->branch_bitmap[MAX_BRANCHES / 64] &= (1ULL << (MAX_BRANCHES % 64)) - 1;
}

static int
bitmap_alloc(uint64_t *bitmap, int n_words, int n_total, uint32_t *out_id)
{
    for (int w = 0; w < n_words; w++) {
        uint64_t bits = bitmap[w];
        if (bits) {
            int b = __builtin_ctzll(bits);
            uint32_t id = (uint32_t)(w * 64 + b);
            bitmap[w] &= ~(1ULL << b);
            *out_id = id;
            return 0;
        }
    }
    return -1;
}

static void
bitmap_free(uint64_t *bitmap, uint32_t id)
{
    int w = id / 64;
    int b = id % 64;
    bitmap[w] |= (1ULL << b);
}

int
kv_prefix_register(KVPrefixTable *t, int64_t kv_offset,
                   uint32_t token_len, uint32_t nblocks,
                   uint32_t *out_id)
{
    uint32_t id;
    if (bitmap_alloc(t->prefix_bitmap,
        (int)(sizeof(t->prefix_bitmap) / sizeof(t->prefix_bitmap[0])),
        MAX_PREFIXES, &id) != 0)
        return -1;
    KVPrefix *p = &t->prefixes[id];
    p->prefix_id      = id;
    p->refcnt         = 1;
    p->kv_block_offset = kv_offset;
    p->token_len       = token_len;
    p->kv_block_count  = nblocks;
    p->flags           = 0;
    *out_id = id;
    return 0;
}

int
kv_prefix_acquire(KVPrefixTable *t, uint32_t prefix_id)
{
    if (prefix_id >= MAX_PREFIXES) return -1;
    KVPrefix *p = &t->prefixes[prefix_id];
    if (p->refcnt == 0) return -1;
    p->refcnt++;
    return 0;
}

int
kv_prefix_release(KVPrefixTable *t, uint32_t prefix_id)
{
    if (prefix_id >= MAX_PREFIXES) return -1;
    KVPrefix *p = &t->prefixes[prefix_id];
    if (p->refcnt == 0) return -1;
    p->refcnt--;
    if (p->refcnt == 0) {
        p->kv_block_offset = -1;
        p->token_len = 0;
        bitmap_free(t->prefix_bitmap, prefix_id);
    }
    return 0;
}

KVPrefix *
kv_prefix_get(KVPrefixTable *t, uint32_t prefix_id)
{
    if (prefix_id >= MAX_PREFIXES) return NULL;
    return &t->prefixes[prefix_id];
}

int
kv_branch_alloc(KVPrefixTable *t, uint32_t rollout_id,
                uint32_t prefix_id, int64_t delta_kv_offset,
                uint32_t delta_len, uint32_t delta_blocks,
                uint32_t *out_branch_id)
{
    uint32_t id;
    if (bitmap_alloc(t->branch_bitmap,
        (int)(sizeof(t->branch_bitmap) / sizeof(t->branch_bitmap[0])),
        MAX_BRANCHES, &id) != 0)
        return -1;
    KVBranch *b = &t->branches[id];
    b->branch_id        = id;
    b->rollout_id       = rollout_id;
    b->prefix_id        = prefix_id;
    b->delta_kv_offset  = delta_kv_offset;
    b->delta_len        = delta_len;
    b->delta_block_count= delta_blocks;
    b->state            = 0;
    b->flags            = 0;
    kv_prefix_acquire(t, prefix_id);
    *out_branch_id = id;
    return 0;
}

int
kv_branch_free(KVPrefixTable *t, uint32_t branch_id)
{
    if (branch_id >= MAX_BRANCHES) return -1;
    KVBranch *b = &t->branches[branch_id];
    kv_prefix_release(t, b->prefix_id);
    b->rollout_id = UINT32_MAX;
    bitmap_free(t->branch_bitmap, branch_id);
    return 0;
}

KVBranch *
kv_branch_get(KVPrefixTable *t, uint32_t branch_id)
{
    if (branch_id >= MAX_BRANCHES) return NULL;
    return &t->branches[branch_id];
}

int64_t
kv_branch_total_offset(KVPrefixTable *t, uint32_t branch_id)
{
    if (branch_id >= MAX_BRANCHES) return -1;
    KVBranch *b = &t->branches[branch_id];
    KVPrefix *p = kv_prefix_get(t, b->prefix_id);
    if (!p) return b->delta_kv_offset;
    return p->kv_block_offset;
}
