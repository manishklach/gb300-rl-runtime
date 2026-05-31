#include "query_producer.h"
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ static void
query_prepare_kernel(float *hidden_buf,
                     __half *query_buf,
                     const __half *proj_buf,
                     uint32_t slots,
                     uint64_t seq_id,
                     uint32_t step,
                     uint32_t slot)
{
    const uint32_t d = threadIdx.x;
    if (d >= QUERY_MODEL_DIM)
        return;

    const uint32_t bounded_slot = slots == 0 ? 0U : (slot & (slots - 1U));
    float *hidden = hidden_buf + bounded_slot * QUERY_MODEL_DIM;
    __half *query = query_buf + bounded_slot * QUERY_MODEL_DIM;

    /*
     * Tiny explicit state producer:
     * - synthesize/update one hidden-state row for this slot
     * - apply a fixed QUERY_MODEL_DIM x QUERY_MODEL_DIM projection
     *
     * This is still deterministic scaffolding, but it now looks like a
     * minimal model-state -> query transformation rather than a host-side
     * direct fill of q vectors.
     */
    const float hidden_val =
        (float)(((int)(((seq_id + 1ULL) * (d + 11U)) + step * 13U) % 43) - 21) / 16.0f;
    hidden[d] = hidden_val;

    float acc = 0.0f;
    for (uint32_t k = 0; k < QUERY_MODEL_DIM; k++) {
        const __half w = proj_buf[k * QUERY_MODEL_DIM + d];
        acc += hidden[k] * __half2float(w);
    }
    query[d] = __float2half_rn(acc);
}

int
query_producer_init(float **d_hidden_buf,
                    __half **d_proj_buf,
                    uint32_t slots)
{
    const size_t hidden_bytes =
        (size_t)slots * QUERY_MODEL_DIM * sizeof(float);
    const size_t proj_bytes =
        (size_t)QUERY_MODEL_DIM * QUERY_MODEL_DIM * sizeof(__half);
    __half *h_proj = (__half *)malloc(proj_bytes);
    if (!h_proj)
        return -1;

    for (uint32_t r = 0; r < QUERY_MODEL_DIM; r++) {
        for (uint32_t c = 0; c < QUERY_MODEL_DIM; c++) {
            const float v =
                (float)(((int)((r + 3U) * (c + 5U)) % 29) - 14) / 32.0f;
            h_proj[r * QUERY_MODEL_DIM + c] = __float2half_rn(v);
        }
    }

    if (cudaMalloc(d_hidden_buf, hidden_bytes) != cudaSuccess) {
        free(h_proj);
        return -1;
    }
    if (cudaMalloc(d_proj_buf, proj_bytes) != cudaSuccess) {
        cudaFree(*d_hidden_buf);
        *d_hidden_buf = NULL;
        free(h_proj);
        return -1;
    }

    cudaMemset(*d_hidden_buf, 0, hidden_bytes);
    cudaMemcpy(*d_proj_buf, h_proj, proj_bytes, cudaMemcpyHostToDevice);
    free(h_proj);
    return 0;
}

void
query_producer_destroy(float *d_hidden_buf,
                       __half *d_proj_buf)
{
    if (d_hidden_buf)
        cudaFree(d_hidden_buf);
    if (d_proj_buf)
        cudaFree(d_proj_buf);
}

int
query_producer_prepare_slot(float *d_hidden_buf,
                            __half *d_query_buf,
                            const __half *d_proj_buf,
                            uint32_t slots,
                            uint64_t seq_id,
                            uint32_t step,
                            uint32_t slot)
{
    query_prepare_kernel<<<1, QUERY_MODEL_DIM>>>(
        d_hidden_buf, d_query_buf, d_proj_buf, slots, seq_id, step, slot);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
