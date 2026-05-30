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
    IdRing    queues[Q_COUNT];
    rollout_slab_t slab;
} RolloutPipeline;

int  pipeline_init(RolloutPipeline *p);
int  pipeline_push(RolloutPipeline *p, pipeline_q_t q, uint32_t rollout_id);
int  pipeline_pop(RolloutPipeline *p, pipeline_q_t q, uint32_t *out_id);
int  pipeline_transition(RolloutPipeline *p, uint32_t rollout_id,
                         rollout_state_t from, rollout_state_t to);

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
