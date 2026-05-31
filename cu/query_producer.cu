#include "query_producer.h"
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ static void
query_prepare_kernel(float *hidden_buf,
                     __half *query_buf,
                     const __half *proj_buf,
                     uint32_t slots,
                     uint32_t slot)
{
    const uint32_t d = threadIdx.x;
    if (d >= QUERY_MODEL_DIM)
        return;

    const uint32_t bounded_slot = slots == 0 ? 0U : (slot & (slots - 1U));
    float *hidden = hidden_buf + bounded_slot * QUERY_MODEL_DIM;
    __half *query = query_buf + bounded_slot * QUERY_MODEL_DIM;

    float acc = 0.0f;
    for (uint32_t k = 0; k < QUERY_MODEL_DIM; k++) {
        const __half w = proj_buf[k * QUERY_MODEL_DIM + d];
        acc += hidden[k] * __half2float(w);
    }
    query[d] = __float2half_rn(acc);
}

int
query_producer_init(__half **d_proj_buf)
{
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

    if (cudaMalloc(d_proj_buf, proj_bytes) != cudaSuccess) {
        free(h_proj);
        return -1;
    }

    cudaMemcpy(*d_proj_buf, h_proj, proj_bytes, cudaMemcpyHostToDevice);
    free(h_proj);
    return 0;
}

void
query_producer_destroy(__half *d_proj_buf)
{
    if (d_proj_buf)
        cudaFree(d_proj_buf);
}

int
query_producer_prepare_slot(const float *d_hidden_buf,
                            __half *d_query_buf,
                            const __half *d_proj_buf,
                            uint32_t slots,
                            uint32_t slot)
{
    query_prepare_kernel<<<1, QUERY_MODEL_DIM>>>(
        (float *)d_hidden_buf, d_query_buf, d_proj_buf, slots, slot);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
