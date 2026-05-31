#include "request.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void rollout_worker(RequestRing*, DoneRing*,
                                GpuRolloutState*, uint64_t*);

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    int n_rollouts = 10000;
    int tokens_per = 128;
    int dev_id = 0;

    int opt;
    while ((opt = getopt(argc, argv, "r:t:d:")) != -1) {
        switch (opt) {
        case 'r': n_rollouts    = atoi(optarg); break;
        case 't': tokens_per    = atoi(optarg); break;
        case 'd': dev_id        = atoi(optarg); break;
        }
    }

    printf("GB300 RL Runtime — GPU-Resident Scheduler Benchmark\n");
    printf("  Rollouts:         %d\n", n_rollouts);
    printf("  Tokens/rollout:   %d\n", tokens_per);
    printf("  Total tokens:     %d\n", n_rollouts * tokens_per);
    printf("  Device:           %d\n\n", dev_id);

    cudaSetDevice(dev_id);

    RequestRing *req_ring;
    DoneRing    *done_ring;
    cudaHostAlloc(&req_ring, sizeof(RequestRing), cudaHostAllocMapped);
    cudaHostAlloc(&done_ring, sizeof(DoneRing), cudaHostAllocMapped);
    memset(req_ring, 0, sizeof(RequestRing));
    memset(done_ring, 0, sizeof(DoneRing));

    GpuRolloutState *d_state;
    uint64_t *d_step_count;
    cudaMalloc(&d_state, sizeof(GpuRolloutState));
    cudaMalloc(&d_step_count, sizeof(uint64_t));
    cudaMemset(d_state, 0, sizeof(GpuRolloutState));
    cudaMemset(d_step_count, 0, sizeof(uint64_t));

    rollout_worker<<<1, 32>>>(req_ring, done_ring, d_state, d_step_count);

    uint64_t t0 = now_ns();

    for (int i = 0; i < n_rollouts; i++) {
        uint32_t pos;
        do {
            pos = req_ring_acquire(req_ring);
        } while (pos == UINT32_MAX);

        RolloutRequest req;
        req.request_id  = i + 1;
        req.max_tokens  = tokens_per;
        req.kv_blocks   = 4;
        req.temperature = 1.0f;
        req.top_k       = 50;
        req.top_p       = 0.9f;
        req.rng_seed    = 42 + (uint64_t)i;
        req_ring->slots[pos] = req;
        req_ring_commit(req_ring);
    }

    uint64_t total_tokens = (uint64_t)n_rollouts * tokens_per;
    uint64_t completed = 0;
    while (completed < (uint64_t)n_rollouts) {
        RolloutDone done;
        while (done_ring_pop(done_ring, &done))
            completed++;
    }

    uint64_t t1 = now_ns();
    uint64_t wall_ns = t1 - t0;

    uint64_t steps;
    cudaMemcpy(&steps, d_step_count, sizeof(steps), cudaMemcpyDeviceToHost);

    double wall_s = wall_ns / 1.0e9;
    printf("── GPU Scheduler Results ──\n");
    printf("  Wall time:          %.3f s\n", wall_s);
    printf("  Throughput:         %.0f rollouts/s\n", n_rollouts / wall_s);
    printf("  Throughput:         %.0f tokens/s\n", total_tokens / wall_s);
    printf("  GPU steps recorded: %lu\n", (unsigned long)steps);
    printf("  CPU dispatches:     %d  (one request submitted per rollout)\n",
           n_rollouts);
    printf("  CPU polls:          %lu  (done notifications)\n",
           (unsigned long)completed);
    printf("  CPU per-token work: none  (GPU managed all decode steps)\n");
    printf("  Done ring overflows:%u\n", done_ring->overflow.value);

    /* sentinel shutdown */
    uint32_t pos;
    do { pos = req_ring_acquire(req_ring); } while (pos == UINT32_MAX);
    RolloutRequest sentinel;
    memset(&sentinel, 0, sizeof(sentinel));
    sentinel.request_id = (uint64_t)-1;
    req_ring->slots[pos] = sentinel;
    req_ring_commit(req_ring);
    cudaDeviceSynchronize();

    cudaFree(d_state);
    cudaFree(d_step_count);
    cudaFreeHost(req_ring);
    cudaFreeHost(done_ring);

    printf("\nDone.\n");
    return 0;
}
