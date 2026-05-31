#include "model_state.h"
#include <cuda_runtime.h>

__global__ static void
model_state_prepare_kernel(float *hidden_buf,
                           uint32_t slots,
                           uint64_t seq_id,
                           uint32_t step,
                           uint32_t slot)
{
    const uint32_t d = threadIdx.x;
    if (d >= MODEL_STATE_DIM)
        return;

    const uint32_t bounded_slot = slots == 0 ? 0U : (slot & (slots - 1U));
    float *hidden = hidden_buf + bounded_slot * MODEL_STATE_DIM;

    /*
     * Tiny explicit state-update rule:
     * - retain part of the previous hidden state
     * - inject deterministic per-sequence/per-step signal
     *
     * This is still scaffolding, but it is now a separate activation
     * preparation stage rather than being fused into query projection.
     */
    const float prev = hidden[d];
    const float base =
        (float)(((int)(((seq_id + 1ULL) * (d + 11U)) + step * 13U) % 43) - 21) / 16.0f;
    const float mix =
        (float)(((int)(((seq_id + 3ULL) * (d + 7U)) + step * 5U) % 31) - 15) / 64.0f;
    hidden[d] = prev * 0.5f + base * 0.375f + mix;
}

int
model_state_init(float **d_hidden_buf, uint32_t slots)
{
    const size_t hidden_bytes =
        (size_t)slots * MODEL_STATE_DIM * sizeof(float);

    if (cudaMalloc(d_hidden_buf, hidden_bytes) != cudaSuccess)
        return -1;

    cudaMemset(*d_hidden_buf, 0, hidden_bytes);
    return 0;
}

void
model_state_destroy(float *d_hidden_buf)
{
    if (d_hidden_buf)
        cudaFree(d_hidden_buf);
}

int
model_state_prepare_slot(float *d_hidden_buf,
                         uint32_t slots,
                         uint64_t seq_id,
                         uint32_t step,
                         uint32_t slot)
{
    model_state_prepare_kernel<<<1, MODEL_STATE_DIM>>>(
        d_hidden_buf, slots, seq_id, step, slot);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
