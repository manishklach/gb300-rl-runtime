#include "pipeline.h"
#include <string.h>
#include <stdlib.h>

static void
id_ring_init(IdRing *r)
{
    memset(r, 0, sizeof(*r));
}

static int
id_ring_push(IdRing *r, uint32_t id)
{
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_acquire);
    uint32_t t = atomic_load_explicit(&r->cons.tail, memory_order_relaxed);
    if (PIPELINE_RING_SIZE - (h - t) < 1)
        return -1;
    uint32_t pos = h & (PIPELINE_RING_SIZE - 1);
    r->slots[pos] = id;
    atomic_store_explicit(&r->prod.head, h + 1, memory_order_release);
    return 0;
}

static int
id_ring_pop(IdRing *r, uint32_t *out_id)
{
    uint32_t t = atomic_load_explicit(&r->cons.tail, memory_order_acquire);
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_relaxed);
    if (t >= h)
        return -1;
    *out_id = r->slots[t & (PIPELINE_RING_SIZE - 1)];
    atomic_store_explicit(&r->cons.tail, t + 1, memory_order_release);
    return 0;
}

int
pipeline_init(RolloutPipeline *p)
{
    memset(p, 0, sizeof(*p));
    for (int i = 0; i < Q_COUNT; i++)
        id_ring_init(&p->queues[i]);
    rollout_slab_init(&p->slab);
    return 0;
}

int
pipeline_push(RolloutPipeline *p, pipeline_q_t q, uint32_t rollout_id)
{
    if (q < 0 || q >= Q_COUNT) return -1;
    return id_ring_push(&p->queues[q], rollout_id);
}

int
pipeline_pop(RolloutPipeline *p, pipeline_q_t q, uint32_t *out_id)
{
    if (q < 0 || q >= Q_COUNT) return -1;
    return id_ring_pop(&p->queues[q], out_id);
}

int
pipeline_transition(RolloutPipeline *p, uint32_t rollout_id,
                    rollout_state_t from, rollout_state_t to)
{
    rollout_t *r = rollout_get(&p->slab, rollout_id);
    if (!r) return -1;
    return rollout_transition(r, from, to);
}
