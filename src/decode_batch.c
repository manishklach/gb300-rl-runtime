#include "decode_batch.h"

void
decode_batch_reset(DecodeDispatchBatch *batch)
{
    if (!batch)
        return;
    batch->count = 0;
}

int
decode_batch_push(DecodeDispatchBatch *batch, const Descriptor *desc)
{
    if (!batch || !desc || batch->count >= DECODE_DESCRIPTOR_BATCH_LIMIT)
        return -1;
    batch->descs[batch->count++] = *desc;
    return 0;
}

int
decode_batch_submit(CommandRing *ring, const DecodeDispatchBatch *batch)
{
    uint32_t pos;
    uint32_t i;

    if (!ring || !batch || batch->count == 0)
        return -1;

    pos = ring_acquire(ring, batch->count);
    if (pos == UINT32_MAX)
        return -1;

    for (i = 0; i < batch->count; i++)
        ring->slots[(pos + i) & (RING_SIZE - 1U)] = batch->descs[i];

    ring_commit(ring, batch->count);
    return 0;
}
