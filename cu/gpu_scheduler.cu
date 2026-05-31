#include "request.h"
#include <cuda_runtime.h>

__device__ static int
find_free_slot(GpuRolloutState *st)
{
    for (int w = 0; w < (int)(sizeof(st->bitmap) / sizeof(st->bitmap[0])); w++) {
        uint64_t bits = ~st->bitmap[w];
        if (bits) {
            int b = __ffsll(bits) - 1;
            st->bitmap[w] |= (1ULL << b);
            return w * 64 + b;
        }
    }
    return -1;
}

__device__ static void
release_slot(GpuRolloutState *st, int idx)
{
    int w = idx / 64;
    int b = idx % 64;
    st->bitmap[w] &= ~(1ULL << b);
}

__global__ void
rollout_worker(RequestRing *req_ring, DoneRing *done_ring,
               GpuRolloutState *state, uint64_t *step_count)
{
    uint32_t tid = threadIdx.x;
    uint32_t warp = blockIdx.x;

    if (tid == 0) {
        while (true) {
            /* poll for new requests */
            RolloutRequest req;
            if (req_ring_consume(req_ring, &req)) {
                if (req.request_id == (uint64_t)-1)
                    return;
                int slot = find_free_slot(state);
                if (slot >= 0) {
                    GpuRolloutSlot *rs = &state->slots[slot];
                    rs->active = 1;
                    rs->request_id = (uint32_t)(req.request_id & 0xFFFFFFFF);
                    rs->tokens_generated = 0;
                    rs->max_tokens = req.max_tokens;
                    rs->kv_blocks = req.kv_blocks;
                    rs->rng_state[0] = req.rng_seed;
                    rs->rng_state[1] = req.rng_seed ^ 0x9e3779b97f4a7c15ULL;
                    rs->rng_state[2] = (req.rng_seed << 17) ^ 0x3c6ef372fe94f82aULL;
                    rs->rng_state[3] = ~req.rng_seed;
                    rs->temperature = req.temperature;
                    rs->top_k = req.top_k;
                    rs->top_p = req.top_p;
                }
            }

            /* process active rollouts (each slot gets one step) */
            for (int s = 0; s < GPU_MAX_ROLLOUTS; s++) {
                GpuRolloutSlot *rs = &state->slots[s];
                if (!rs->active) continue;
                if (rs->tokens_generated >= rs->max_tokens) {
                    rs->active = 0;
                    RolloutDone done;
                    done.request_id = rs->request_id;
                    done.rollout_id = (uint32_t)s;
                    done.tokens_generated = rs->tokens_generated;
                    done.reward = (float)(rs->tokens_generated & 0xFF) / 255.0f;
                    done.status = 0;
                    while (done_ring_push(done_ring, &done) != 0)
                        __nanosleep(100);
                    release_slot(state, s);
                    if (step_count)
                        atomicAdd((unsigned long long *)step_count, 1ULL);
                    continue;
                }
                rs->tokens_generated++;
            }

            __nanosleep(100);
        }
    }
}
