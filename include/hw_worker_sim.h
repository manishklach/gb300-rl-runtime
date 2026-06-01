#pragma once

#include "hw_ring.h"
#include <stdint.h>

typedef struct {
    hw_ring_t *cmdq;
    hw_ring_t *doneq;
    volatile uint32_t *stop;
    uint64_t decoded_tokens;
    uint64_t completions;
} hw_worker_sim_t;

void hw_worker_sim_run(hw_worker_sim_t *worker);
