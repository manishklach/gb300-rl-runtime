#include "model_state.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

__global__ static void
model_state_prepare_kernel(float *hidden_buf,
                           const __half *input_proj,
                           const __half *residual_proj,
                           const float *bias,
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
    __shared__ float prev_hidden[MODEL_STATE_DIM];
    __shared__ float input_signal[MODEL_STATE_DIM];

    /*
     * Tiny explicit residual block:
     * - synthesize an input activation vector
     * - project that vector and the previous hidden state
     * - apply a smooth nonlinearity
     * - write back a residual-style hidden update
     *
     * This is still scaffolding, but it is now a separate activation
     * block with explicit weights rather than a hand-written scalar mix.
     */
    prev_hidden[d] = hidden[d];
    input_signal[d] =
        (float)(((int)(((seq_id + 1ULL) * (d + 11U)) + step * 13U) % 43) - 21) / 16.0f;
    __syncthreads();

    float mlp_acc = bias[d];
    float res_acc = 0.0f;
    for (uint32_t k = 0; k < MODEL_STATE_DIM; k++) {
        mlp_acc += input_signal[k] *
                   __half2float(input_proj[k * MODEL_STATE_DIM + d]);
        res_acc += prev_hidden[k] *
                   __half2float(residual_proj[k * MODEL_STATE_DIM + d]);
    }

    const float gated = mlp_acc / (1.0f + expf(-mlp_acc));
    hidden[d] = prev_hidden[d] * 0.5f + res_acc * 0.125f +
                input_signal[d] * 0.25f + gated * 0.125f;
}

int
model_state_init(ModelStateBuffers *state, uint32_t slots)
{
    const size_t hidden_bytes =
        (size_t)slots * MODEL_STATE_DIM * sizeof(float);
    const size_t proj_bytes =
        (size_t)MODEL_STATE_DIM * MODEL_STATE_DIM * sizeof(__half);
    const size_t bias_bytes =
        (size_t)MODEL_STATE_DIM * sizeof(float);
    __half *h_input_proj = (__half *)malloc(proj_bytes);
    __half *h_residual_proj = (__half *)malloc(proj_bytes);
    float *h_bias = (float *)malloc(bias_bytes);

    if (!state || !h_input_proj || !h_residual_proj || !h_bias) {
        free(h_input_proj);
        free(h_residual_proj);
        free(h_bias);
        return -1;
    }

    for (uint32_t r = 0; r < MODEL_STATE_DIM; r++) {
        h_bias[r] =
            (float)(((int)((r + 5U) * 7U) % 19) - 9) / 64.0f;
        for (uint32_t c = 0; c < MODEL_STATE_DIM; c++) {
            const float in_v =
                (float)(((int)((r + 3U) * (c + 1U)) % 23) - 11) / 48.0f;
            const float res_v =
                (float)(((int)((r + 9U) * (c + 5U)) % 17) - 8) / 64.0f;
            h_input_proj[r * MODEL_STATE_DIM + c] = __float2half_rn(in_v);
            h_residual_proj[r * MODEL_STATE_DIM + c] = __float2half_rn(res_v);
        }
    }

    if (cudaMalloc(&state->hidden_buf, hidden_bytes) != cudaSuccess) {
        free(h_input_proj);
        free(h_residual_proj);
        free(h_bias);
        return -1;
    }
    if (cudaMalloc(&state->input_proj, proj_bytes) != cudaSuccess) {
        cudaFree(state->hidden_buf);
        state->hidden_buf = NULL;
        free(h_input_proj);
        free(h_residual_proj);
        free(h_bias);
        return -1;
    }
    if (cudaMalloc(&state->residual_proj, proj_bytes) != cudaSuccess) {
        cudaFree(state->hidden_buf);
        cudaFree(state->input_proj);
        state->hidden_buf = NULL;
        state->input_proj = NULL;
        free(h_input_proj);
        free(h_residual_proj);
        free(h_bias);
        return -1;
    }
    if (cudaMalloc(&state->bias, bias_bytes) != cudaSuccess) {
        cudaFree(state->hidden_buf);
        cudaFree(state->input_proj);
        cudaFree(state->residual_proj);
        state->hidden_buf = NULL;
        state->input_proj = NULL;
        state->residual_proj = NULL;
        free(h_input_proj);
        free(h_residual_proj);
        free(h_bias);
        return -1;
    }

    cudaMemset(state->hidden_buf, 0, hidden_bytes);
    cudaMemcpy(state->input_proj, h_input_proj, proj_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(state->residual_proj, h_residual_proj, proj_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(state->bias, h_bias, bias_bytes, cudaMemcpyHostToDevice);

    free(h_input_proj);
    free(h_residual_proj);
    free(h_bias);
    return 0;
}

void
model_state_destroy(ModelStateBuffers *state)
{
    if (!state)
        return;
    if (state->hidden_buf)
        cudaFree(state->hidden_buf);
    if (state->input_proj)
        cudaFree(state->input_proj);
    if (state->residual_proj)
        cudaFree(state->residual_proj);
    if (state->bias)
        cudaFree(state->bias);
    state->hidden_buf = NULL;
    state->input_proj = NULL;
    state->residual_proj = NULL;
    state->bias = NULL;
}

int
model_state_prepare_slot(ModelStateBuffers *state,
                         uint32_t slots,
                         uint64_t seq_id,
                         uint32_t step,
                         uint32_t slot)
{
    model_state_prepare_kernel<<<1, MODEL_STATE_DIM>>>(
        state->hidden_buf, state->input_proj, state->residual_proj,
        state->bias, slots, seq_id, step, slot);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
