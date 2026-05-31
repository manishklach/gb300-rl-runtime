#pragma once

#include <stddef.h>
#include <stdint.h>

/*
 * v0.2.2-a fixed-shape KV layout scaffold.
 *
 * This header intentionally defines one concrete decode-oriented layout
 * rather than a configurable runtime schema.  The goal is to establish
 * explicit hardware-facing invariants before the first real attention
 * microkernel lands.
 */

#define KV_LAYOUT_HEAD_DIM         128U
#define KV_LAYOUT_TOKENS_PER_BLOCK 32U
#define KV_LAYOUT_SCALAR_BYTES     2U   /* fp16/bf16-sized lane */
#define KV_LAYOUT_VEC_BYTES        16U  /* 128-bit vector load target */
#define KV_LAYOUT_ALIGNMENT        128U

#define KV_LAYOUT_K_BYTES \
    (KV_LAYOUT_TOKENS_PER_BLOCK * KV_LAYOUT_HEAD_DIM * KV_LAYOUT_SCALAR_BYTES)
#define KV_LAYOUT_V_BYTES KV_LAYOUT_K_BYTES
#define KV_LAYOUT_BLOCK_BYTES (KV_LAYOUT_K_BYTES + KV_LAYOUT_V_BYTES)

typedef struct {
    uint32_t token_count;
    uint32_t head_dim;
    uint32_t scalar_bytes;
    uint32_t vec_bytes;
    uint32_t k_stride_bytes;
    uint32_t v_stride_bytes;
    uint32_t block_bytes;
    uint32_t alignment;
} KvLayoutDesc;

_Static_assert((KV_LAYOUT_HEAD_DIM * KV_LAYOUT_SCALAR_BYTES) % KV_LAYOUT_VEC_BYTES == 0,
               "Head dimension must support vectorized loads");
_Static_assert(KV_LAYOUT_BLOCK_BYTES % KV_LAYOUT_ALIGNMENT == 0,
               "KV block must preserve alignment");

static inline KvLayoutDesc
kv_layout_desc_fixed128(void)
{
    KvLayoutDesc desc;
    desc.token_count = KV_LAYOUT_TOKENS_PER_BLOCK;
    desc.head_dim = KV_LAYOUT_HEAD_DIM;
    desc.scalar_bytes = KV_LAYOUT_SCALAR_BYTES;
    desc.vec_bytes = KV_LAYOUT_VEC_BYTES;
    desc.k_stride_bytes = KV_LAYOUT_HEAD_DIM * KV_LAYOUT_SCALAR_BYTES;
    desc.v_stride_bytes = KV_LAYOUT_HEAD_DIM * KV_LAYOUT_SCALAR_BYTES;
    desc.block_bytes = KV_LAYOUT_BLOCK_BYTES;
    desc.alignment = KV_LAYOUT_ALIGNMENT;
    return desc;
}

static inline uint32_t
kv_layout_block_offset_bytes(uint32_t block_idx)
{
    return block_idx * KV_LAYOUT_BLOCK_BYTES;
}

static inline uint32_t
kv_layout_k_offset_bytes(uint32_t token_idx)
{
    return token_idx * (KV_LAYOUT_HEAD_DIM * KV_LAYOUT_SCALAR_BYTES);
}

static inline uint32_t
kv_layout_v_plane_base_bytes(void)
{
    return KV_LAYOUT_K_BYTES;
}

static inline uint32_t
kv_layout_v_offset_bytes(uint32_t token_idx)
{
    return kv_layout_v_plane_base_bytes() +
           token_idx * (KV_LAYOUT_HEAD_DIM * KV_LAYOUT_SCALAR_BYTES);
}
