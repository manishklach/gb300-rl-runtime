#pragma once

#include <stdint.h>

#define DESC_OP_DECODE       1U
#define DESC_OP_REWARD       2U
#define DESC_OP_PREFILL      3U
#define DESC_OP_STOP         255U

#define DESC_FLAG_NEEDS_REWARD (1u << 0)
#define DESC_FLAG_DONE         (1u << 1)
#define DESC_FLAG_COW_PREFIX   (1u << 2)

typedef struct __attribute__((packed, aligned(64))) {
    uint16_t opcode;
    uint16_t flags;
    uint32_t rollout_id;

    uint32_t kv_arena_id;
    uint32_t prefix_id;

    uint64_t kv_offset;
    uint64_t delta_offset;

    uint32_t seq_len;
    uint32_t max_tokens;

    uint32_t reward_model_id;
    uint32_t reserved0;

    uint64_t user_data;
    uint64_t checksum_or_cookie;
} hw_desc_t;

_Static_assert(sizeof(hw_desc_t) == 64, "hw_desc_t must be one cache line");
