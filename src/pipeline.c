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
    p->policy = SCHED_FIFO;
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

void
pipeline_credits_set(RolloutPipeline *p, uint32_t max_decode,
                     uint32_t max_reward, uint32_t max_trajectory,
                     uint32_t kv_limit)
{
    p->credits.max_decode_credits     = max_decode;
    p->credits.max_reward_credits     = max_reward;
    p->credits.max_trajectory_credits = max_trajectory;
    p->credits.kv_block_limit         = kv_limit;
}

int
pipeline_try_push(RolloutPipeline *p, pipeline_q_t q, uint32_t rollout_id)
{
    PipelineCredits *c = &p->credits;
    switch (q) {
    case Q_DECODE:
        if (c->max_decode_credits && c->decode_used >= c->max_decode_credits)
            return -1;
        if (pipeline_push(p, q, rollout_id) != 0) return -1;
        c->decode_used++;
        return 0;
    case Q_REWARD:
        if (c->max_reward_credits && c->reward_used >= c->max_reward_credits)
            return -1;
        if (pipeline_push(p, q, rollout_id) != 0) return -1;
        c->reward_used++;
        return 0;
    case Q_TRAJECTORY:
        if (c->max_trajectory_credits && c->trajectory_used >= c->max_trajectory_credits)
            return -1;
        if (pipeline_push(p, q, rollout_id) != 0) return -1;
        c->trajectory_used++;
        return 0;
    default:
        return pipeline_push(p, q, rollout_id);
    }
}

void
pipeline_release(RolloutPipeline *p, pipeline_q_t q, uint32_t n)
{
    PipelineCredits *c = &p->credits;
    switch (q) {
    case Q_DECODE:
        if (n > c->decode_used) c->decode_used = 0;
        else c->decode_used -= n;
        break;
    case Q_REWARD:
        if (n > c->reward_used) c->reward_used = 0;
        else c->reward_used -= n;
        break;
    case Q_TRAJECTORY:
        if (n > c->trajectory_used) c->trajectory_used = 0;
        else c->trajectory_used -= n;
        break;
    default:
        break;
    }
}

int
pipeline_set_schedule_policy(RolloutPipeline *p, SchedulePolicy policy)
{
    if (policy != SCHED_FIFO)
        return -1;
    p->policy = policy;
    return 0;
}

int
pipeline_schedule(RolloutPipeline *p, pipeline_q_t q, uint32_t *out_id)
{
    if (q < 0 || q >= Q_COUNT) return -1;

    if (p->policy == SCHED_FIFO)
        return pipeline_pop(p, q, out_id);

    /* ── aspirational non-FIFO policies ──────────────────────────────
     * SCHED_SHORTEST_REMAINING and SCHED_PREFIX_SHARING are NOT
     * safe in v0.4.x: the search-then-advance-to-best_idx pattern
     * in the original implementation skipped every entry between
     * the current tail and best_idx, silently dropping queued
     * rollouts.
     *
     * Restore after implementing a scheduler that can remove
     * arbitrary entries without losing queued work.  The original
     * code is preserved below as a reference algorithm. */
#if 0
    IdRing *r = &p->queues[q];
    uint32_t t = atomic_load_explicit(&r->cons.tail, memory_order_acquire);
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_relaxed);
    if (t >= h) return -1;

    uint32_t count = h - t;
    uint32_t best_idx = UINT32_MAX;
    int64_t  best_val = INT64_MAX;

    for (uint32_t i = 0; i < count && i < 64; i++) {
        uint32_t id = r->slots[(t + i) & (PIPELINE_RING_SIZE - 1)];
        rollout_t *ro = rollout_get(&p->slab, id);
        if (!ro) continue;

        int64_t val = 0;
        switch (p->policy) {
        case SCHED_SHORTEST_REMAINING:
            val = (int64_t)(ro->max_tokens - ro->seq_len);
            break;
        case SCHED_PREFIX_SHARING:
            val = (int64_t)ro->rollout_id;
            break;
        default:
            val = i;
            break;
        }

        if (val < best_val) {
            best_val = val;
            best_idx = t + i;
        }
    }

    if (best_idx == UINT32_MAX)
        return id_ring_pop(r, out_id);

    uint32_t pos = best_idx & (PIPELINE_RING_SIZE - 1);
    *out_id = r->slots[pos];

    uint32_t new_tail = best_idx + 1;
    atomic_store_explicit(&r->cons.tail, new_tail, memory_order_release);

    return 0;
#else
    (void)out_id;
    return -1;
#endif
}

uint32_t
pipeline_occupancy(const RolloutPipeline *p, pipeline_q_t q)
{
    if (q < 0 || q >= Q_COUNT) return 0;
    const IdRing *r = &p->queues[q];
    uint32_t t = atomic_load_explicit(&r->cons.tail, memory_order_acquire);
    uint32_t h = atomic_load_explicit(&r->prod.head, memory_order_relaxed);
    return h - t;
}

uint32_t
pipeline_stage_available_credits(const RolloutPipeline *p, pipeline_q_t q)
{
    if (!p)
        return 0;

    const PipelineCredits *c = &p->credits;
    switch (q) {
    case Q_DECODE:
        return c->max_decode_credits > c->decode_used ?
               (c->max_decode_credits - c->decode_used) : 0U;
    case Q_REWARD:
        return c->max_reward_credits > c->reward_used ?
               (c->max_reward_credits - c->reward_used) : 0U;
    case Q_TRAJECTORY:
        return c->max_trajectory_credits > c->trajectory_used ?
               (c->max_trajectory_credits - c->trajectory_used) : 0U;
    default:
        return UINT32_MAX;
    }
}

uint32_t
pipeline_stage_target_batch(const RolloutPipeline *p, pipeline_q_t q,
                            uint32_t max_batch)
{
    uint32_t occ = pipeline_occupancy(p, q);
    uint32_t credit = pipeline_stage_available_credits(p, q);
    uint32_t limit = max_batch;

    if (credit != UINT32_MAX && credit < limit)
        limit = credit;
    if (occ < limit)
        limit = occ;
    return limit;
}

void
pipeline_snapshot(const RolloutPipeline *p, PipelineSnapshot *snap)
{
    if (!p || !snap)
        return;

    memset(snap, 0, sizeof(*snap));
    for (int q = 0; q < Q_COUNT; q++) {
        snap->queue_occupancy[q] = pipeline_occupancy(p, (pipeline_q_t)q);
        snap->stage_credit_headroom[q] =
            pipeline_stage_available_credits(p, (pipeline_q_t)q);
    }
    snap->credits = p->credits;
    snap->policy = p->policy;
}
