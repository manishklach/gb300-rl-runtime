#include "infer_submit.h"

int
infer_submit_decode(infer_hw_ctx_t *ctx,
                    uint32_t rollout_id,
                    uint64_t kv_offset,
                    uint64_t delta_offset,
                    uint32_t prefix_id,
                    uint32_t seq_len,
                    uint32_t max_tokens)
{
    hw_desc_t desc;

    if (!ctx || !ctx->cmdq || !ctx->doorbell)
        return -1;

    desc.opcode = DESC_OP_DECODE;
    desc.flags = prefix_id ? DESC_FLAG_COW_PREFIX : 0u;
    desc.rollout_id = rollout_id;
    desc.kv_arena_id = ctx->kv_arena_id;
    desc.prefix_id = prefix_id;
    desc.kv_offset = kv_offset;
    desc.delta_offset = delta_offset;
    desc.seq_len = seq_len;
    desc.max_tokens = max_tokens;
    desc.reward_model_id = 0u;
    desc.reserved0 = ctx->gpu_group_id;
    desc.user_data = rollout_id;
    desc.checksum_or_cookie = kv_offset ^ delta_offset ^
                              ((uint64_t)rollout_id << 32);

    if (hw_ring_push(ctx->cmdq, &desc) != 0)
        return -1;

    mmio_write32(ctx->doorbell, hw_ring_load_acquire(&ctx->cmdq->tail));
    return 0;
}
