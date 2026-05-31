#pragma once

#include "descriptor.h"
#include "ring.h"
#include <stdint.h>

#define DECODE_DESCRIPTOR_BATCH_LIMIT 32U

typedef struct {
    uint32_t count;
    Descriptor descs[DECODE_DESCRIPTOR_BATCH_LIMIT];
} DecodeDispatchBatch;

void decode_batch_reset(DecodeDispatchBatch *batch);
int decode_batch_push(DecodeDispatchBatch *batch, const Descriptor *desc);
int decode_batch_submit(CommandRing *ring, const DecodeDispatchBatch *batch);
