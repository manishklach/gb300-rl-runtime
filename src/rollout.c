#include "rollout.h"
#include <string.h>

void
rollout_slab_init(rollout_slab_t *slab)
{
    memset(slab, 0, sizeof(*slab));
    for (uint32_t i = 0; i < MAX_ROLLOUTS; i++) {
        slab->slots[i].rollout_id = i;
        slab->slots[i].state = ROLL_FREE;
    }
    slab->free_bitmap[0] = ~0ULL;
    for (int w = 1; w < (int)(sizeof(slab->free_bitmap) / sizeof(slab->free_bitmap[0])); w++)
        slab->free_bitmap[w] = ~0ULL;
    if (MAX_ROLLOUTS % 64)
        slab->free_bitmap[MAX_ROLLOUTS / 64] &= (1ULL << (MAX_ROLLOUTS % 64)) - 1;
}

int
rollout_alloc(rollout_slab_t *slab, uint32_t *out_id)
{
    for (int w = 0; w < (int)(sizeof(slab->free_bitmap) / sizeof(slab->free_bitmap[0])); w++) {
        uint64_t bits = slab->free_bitmap[w];
        if (bits) {
            int b = __builtin_ctzll(bits);
            uint32_t id = (uint32_t)(w * 64 + b);
            slab->free_bitmap[w] &= ~(1ULL << b);
            rollout_t *r = &slab->slots[id];
            memset(r, 0, sizeof(*r));
            r->rollout_id = id;
            r->state = ROLL_FREE;
            *out_id = id;
            return 0;
        }
    }
    return -1;
}

void
rollout_free(rollout_slab_t *slab, uint32_t id)
{
    if (id >= MAX_ROLLOUTS) return;
    rollout_t *r = &slab->slots[id];
    r->state = ROLL_FREE;
    int w = id / 64;
    int b = id % 64;
    slab->free_bitmap[w] |= (1ULL << b);
}

rollout_t *
rollout_get(rollout_slab_t *slab, uint32_t id)
{
    if (id >= MAX_ROLLOUTS) return NULL;
    return &slab->slots[id];
}

int
rollout_transition(rollout_t *r, rollout_state_t from, rollout_state_t to)
{
    if (!rollout_is_valid_transition(from, to))
        return -1;
    uint32_t expected = (uint32_t)from;
    return __atomic_compare_exchange_n(
        &r->state, &expected, (uint32_t)to, 0,
        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
    ) ? 0 : -1;
}
