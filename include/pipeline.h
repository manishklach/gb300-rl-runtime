#pragma once
#include <stdint.h>
#include <stdatomic.h>
#include "rollout.h"

#define PIPELINE_RING_SIZE 4096

typedef struct __attribute__((packed)) {
    volatile uint32_t head __attribute__((aligned(64)));
    uint32_t          tail;
    uint8_t           pad[56];
} IdRingIndex;

typedef struct {
    IdRingIndex prod __attribute__((aligned(64)));
    IdRingIndex cons __attribute__((aligned(64)));
    uint32_t    slots[PIPELINE_RING_SIZE] __attribute__((aligned(64)));
} IdRing;

typedef enum {
    Q_FREE = 0,
    Q_PREFILL,
    Q_DECODE,
    Q_REWARD,
    Q_TRAJECTORY,
    Q_DONE,
    Q_COUNT
} pipeline_q_t;

typedef struct {
    uint32_t max_decode_credits;
    uint32_t max_reward_credits;
    uint32_t max_trajectory_credits;
    uint32_t kv_block_limit;
    uint32_t decode_used;
    uint32_t reward_used;
    uint32_t trajectory_used;
    uint32_t kv_blocks_used;
} PipelineCredits;

typedef enum {
    SCHED_FIFO,
    SCHED_SHORTEST_REMAINING,
    SCHED_PREFIX_SHARING,
} SchedulePolicy;

typedef struct {
    IdRing          queues[Q_COUNT];
    rollout_slab_t  slab;
    PipelineCredits credits;
    SchedulePolicy  policy;
} RolloutPipeline;

int  pipeline_init(RolloutPipeline *p);
int  pipeline_push(RolloutPipeline *p, pipeline_q_t q, uint32_t rollout_id);
int  pipeline_pop(RolloutPipeline *p, pipeline_q_t q, uint32_t *out_id);
int  pipeline_transition(RolloutPipeline *p, uint32_t rollout_id,
                         rollout_state_t from, rollout_state_t to);

void pipeline_credits_set(RolloutPipeline *p, uint32_t max_decode,
                          uint32_t max_reward, uint32_t max_trajectory,
                          uint32_t kv_limit);
int  pipeline_try_push(RolloutPipeline *p, pipeline_q_t q,
                       uint32_t rollout_id);
void pipeline_release(RolloutPipeline *p, pipeline_q_t q, uint32_t n);

void pipeline_set_schedule_policy(RolloutPipeline *p, SchedulePolicy policy);
int  pipeline_schedule(RolloutPipeline *p, pipeline_q_t q, uint32_t *out_id);

uint32_t pipeline_occupancy(const RolloutPipeline *p, pipeline_q_t q);

static inline const char *pipeline_q_name(pipeline_q_t q)
{
    switch (q) {
    case Q_FREE:       return "free";
    case Q_PREFILL:    return "prefill";
    case Q_DECODE:     return "decode";
    case Q_REWARD:     return "reward";
    case Q_TRAJECTORY: return "trajectory";
    case Q_DONE:       return "done";
    default:           return "?";
    }
}

static inline const char *rollout_state_name(rollout_state_t s)
{
    switch (s) {
    case ROLL_FREE:             return "FREE";
    case ROLL_PREFILL_READY:    return "PREFILL_READY";
    case ROLL_DECODING:         return "DECODING";
    case ROLL_REWARD_PENDING:   return "REWARD_PENDING";
    case ROLL_TRAJECTORY_READY: return "TRAJECTORY_READY";
    case ROLL_DONE:             return "DONE";
    default:                    return "?";
    }
}

static inline const char *schedule_policy_name(SchedulePolicy p)
{
    switch (p) {
    case SCHED_FIFO:              return "FIFO";
    case SCHED_SHORTEST_REMAINING: return "SHORTEST_REMAINING";
    case SCHED_PREFIX_SHARING:     return "PREFIX_SHARING";
    default:                      return "?";
    }
}
