#include "cp_async.cuh"
#include "prefetch.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr uint32_t kStageBytes = KV_BLOCK_SIZE;
constexpr int kDefaultBlocks = 4096;
constexpr int kWarmupIters = 10;
constexpr int kMeasureIters = 50;

__global__ void baseline_prefetch_kernel(const uint8_t *src,
                                         uint8_t *sink,
                                         uint32_t blocks)
{
    extern __shared__ uint8_t smem[];
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t block_idx = blockIdx.x;
    if (block_idx >= blocks)
        return;

    const uint8_t *block_src = src + (size_t)block_idx * kStageBytes;
    uint8_t *block_dst = smem;
    for (uint32_t off = lane; off < kStageBytes; off += 32U)
        block_dst[off] = block_src[off];
    __syncthreads();

    for (uint32_t off = lane; off < kStageBytes; off += 32U)
        sink[(size_t)block_idx * kStageBytes + off] = block_dst[off];
}

__device__ void prefetch_copy_async(const uint8_t *src, uint8_t *smem)
{
    const uint32_t lane = threadIdx.x & 31U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (uint32_t off = lane * PREFETCH_CHUNK_BYTES;
         off < kStageBytes;
         off += 32U * PREFETCH_CHUNK_BYTES) {
        cp_async_ca_16(smem + off, src + off);
    }
    cp_async_commit();
    cp_async_wait_all();
#else
    for (uint32_t off = lane; off < kStageBytes; off += 32U)
        smem[off] = src[off];
#endif
    __syncthreads();
}

__global__ void cp_async_prefetch_kernel(const uint8_t *src,
                                         uint8_t *sink,
                                         uint32_t blocks)
{
    extern __shared__ uint8_t smem[];
    const uint32_t block_idx = blockIdx.x;
    if (block_idx >= blocks)
        return;

    const uint32_t lane = threadIdx.x & 31U;
    uint8_t *stage_ptr = smem;
    prefetch_copy_async(src + (size_t)block_idx * kStageBytes, stage_ptr);
    for (uint32_t off = lane; off < kStageBytes; off += 32U)
        sink[(size_t)block_idx * kStageBytes + off] = stage_ptr[off];
}

static void check_cuda(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

} // namespace

int main(int argc, char **argv)
{
    const uint32_t blocks = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10)
                                       : (uint32_t)kDefaultBlocks;
    const size_t total_bytes = (size_t)blocks * kStageBytes;

    uint8_t *src = NULL;
    uint8_t *sink = NULL;
    check_cuda(cudaMalloc(&src, total_bytes), "cudaMalloc src");
    check_cuda(cudaMalloc(&sink, total_bytes), "cudaMalloc sink");
    check_cuda(cudaMemset(src, 0x5A, total_bytes), "cudaMemset src");
    check_cuda(cudaMemset(sink, 0, total_bytes), "cudaMemset sink");

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    dim3 grid(blocks);
    dim3 block(32);
    const size_t smem_bytes = kStageBytes;

    for (int i = 0; i < kWarmupIters; i++) {
        baseline_prefetch_kernel<<<grid, block, smem_bytes>>>(src, sink, blocks);
        cp_async_prefetch_kernel<<<grid, block, smem_bytes>>>(src, sink, blocks);
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start baseline");
    for (int i = 0; i < kMeasureIters; i++)
        baseline_prefetch_kernel<<<grid, block, smem_bytes>>>(src, sink, blocks);
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop baseline");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize baseline");

    float baseline_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&baseline_ms, start, stop),
               "cudaEventElapsedTime baseline");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start cp.async");
    for (int i = 0; i < kMeasureIters; i++)
        cp_async_prefetch_kernel<<<grid, block, smem_bytes>>>(src, sink, blocks);
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop cp.async");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize cp.async");

    float async_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&async_ms, start, stop),
               "cudaEventElapsedTime cp.async");

    const double gib = (double)total_bytes * (double)kMeasureIters / (1024.0 * 1024.0 * 1024.0);
    const double baseline_gibs = gib / ((double)baseline_ms / 1000.0);
    const double async_gibs = gib / ((double)async_ms / 1000.0);

    printf("GB300 RL Runtime - Prefetch Benchmark\n");
    printf("  blocks:              %u\n", blocks);
    printf("  bytes per block:     %u\n", kStageBytes);
    printf("  total bytes/round:   %.2f MiB\n", (double)total_bytes / (1024.0 * 1024.0));
    printf("  measurement rounds:  %d\n\n", kMeasureIters);
    printf("  baseline shared copy:  %.3f ms  (%.2f GiB/s)\n", baseline_ms, baseline_gibs);
    printf("  cp.async staged copy:  %.3f ms  (%.2f GiB/s)\n", async_ms, async_gibs);
    if (baseline_ms > 0.0f)
        printf("  speedup:              %.2fx\n", baseline_ms / async_ms);

    printf("\nInterpretation:\n");
    printf("  This isolates the global->shared KV staging cost, separate from\n");
    printf("  decode math and control-plane overhead. Treat it as a memory-path\n");
    printf("  microbenchmark, not an end-to-end token benchmark.\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(src);
    cudaFree(sink);
    return 0;
}
