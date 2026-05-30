#pragma once
#include <stdint.h>
#include <stddef.h>

#define MAX_ROLLOUTS 8192

typedef enum {
    ROLL_FREE = 0,
    ROLL_PREFILL_READY,
    ROLL_DECODING,
    ROLL_REWARD_PENDING,
    ROLL_TRAJECTORY_READY,
    ROLL_DONE
} rollout_state_t;

typedef struct __attribute__((packed)) {
    uint32_t rollout_id;
    uint32_t state;
    uint32_t kv_arena_id;
    int64_t  kv_block_idx;
    uint32_t seq_len;
    uint32_t max_tokens;
    uint32_t reward_id;
    uint32_t flags;
    uint64_t rng_seed;
    uint32_t pad;
} rollout_t;

_Static_assert(sizeof(rollout_t) == 40, "rollout_t must be 40 bytes");

typedef struct {
    rollout_t slots[MAX_ROLLOUTS];
    uint64_t  free_bitmap[(MAX_ROLLOUTS + 63) / 64];
} rollout_slab_t;

void  rollout_slab_init(rollout_slab_t *slab);
int   rollout_alloc(rollout_slab_t *slab, uint32_t *out_id);
void  rollout_free(rollout_slab_t *slab, uint32_t id);
rollout_t *rollout_get(rollout_slab_t *slab, uint32_t id);
int   rollout_transition(rollout_t *r, rollout_state_t from, rollout_state_t to);

static inline int rollout_is_valid_transition(rollout_state_t from, rollout_state_t to)
{
    switch (from) {
    case ROLL_FREE:               return to == ROLL_PREFILL_READY;
    case ROLL_PREFILL_READY:      return to == ROLL_DECODING;
    case ROLL_DECODING:           return to == ROLL_REWARD_PENDING || to == ROLL_DONE;
    case ROLL_REWARD_PENDING:     return to == ROLL_TRAJECTORY_READY || to == ROLL_DECODING;
    case ROLL_TRAJECTORY_READY:   return to == ROLL_DONE;
    case ROLL_DONE:               return to == ROLL_FREE;
    }
    return 0;
}
