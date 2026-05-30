#include "reward.h"
#include <string.h>
#include <stdlib.h>

void
reward_ring_init(RewardRing *r)
{
    memset(r, 0, sizeof(*r));
}

int
reward_push(RewardRing *r, const RewardDesc *d)
{
    uint32_t h = __atomic_load_n(&r->prod.head, __ATOMIC_ACQUIRE);
    uint32_t t = __atomic_load_n(&r->cons.tail, __ATOMIC_RELAXED);
    if (REWARD_RING_SIZE - (h - t) < 1)
        return -1;
    uint32_t pos = h & (REWARD_RING_SIZE - 1);
    r->slots[pos] = *d;
    __atomic_store_n(&r->prod.head, h + 1, __ATOMIC_RELEASE);
    return 0;
}

int
reward_pop(RewardRing *r, RewardDesc *d)
{
    uint32_t t = __atomic_load_n(&r->cons.tail, __ATOMIC_ACQUIRE);
    uint32_t h = __atomic_load_n(&r->prod.head, __ATOMIC_RELAXED);
    if (t >= h)
        return -1;
    *d = r->slots[t & (REWARD_RING_SIZE - 1)];
    __atomic_store_n(&r->cons.tail, t + 1, __ATOMIC_RELEASE);
    return 0;
}

float
reward_score_mock(const void *tokens, uint32_t n)
{
    (void)tokens;
    return (float)(n & 0xFF) / 255.0f;
}
