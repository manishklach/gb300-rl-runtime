#pragma once
#include <stdint.h>

#define REWARD_RING_SIZE 1024

typedef struct __attribute__((packed)) {
    uint32_t rollout_id;
    uint32_t token_start;
    uint32_t token_count;
    uint32_t reward_model_id;
    float    reward;
    uint32_t flags;
} RewardDesc;

_Static_assert(sizeof(RewardDesc) == 24, "RewardDesc must be 24 bytes");

typedef struct __attribute__((packed)) {
    volatile uint32_t head __attribute__((aligned(64)));
    uint32_t          tail;
    uint8_t           pad[56];
} RewardIndex;

typedef struct {
    RewardIndex prod __attribute__((aligned(64)));
    RewardIndex cons __attribute__((aligned(64)));
    RewardDesc  slots[REWARD_RING_SIZE] __attribute__((aligned(128)));
} RewardRing;

void  reward_ring_init(RewardRing *r);
int   reward_push(RewardRing *r, const RewardDesc *d);
int    reward_pop(RewardRing *r, RewardDesc *d);
float reward_score_mock(const void *tokens, uint32_t n);
