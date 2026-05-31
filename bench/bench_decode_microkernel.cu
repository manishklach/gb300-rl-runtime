#include "attention_decode.h"
#include "descriptor.h"
#include "sample.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ static void
bench_decode_kernel(uint32_t *token_out,
                    uint64_t *cycles_out,
                    uint32_t *bytes_out,
                    uint32_t *tiles_out,
                    const uint8_t *kv_block,
                    SampleState *sample_st,
                    Descriptor desc)
{
    extern __shared__ uint8_t smem[];
    if (threadIdx.x == 0) {
        DecodeStepResult result =
            attention_decode_step_fixed128(&desc, kv_block, sample_st, smem);
        token_out[0] = result.token_id;
        cycles_out[0] = result.cycle_estimate;
        bytes_out[0] = result.bytes_touched;
        tiles_out[0] = result.tile_count;
    }
}

int
main(int argc, char **argv)
{
    int dev_id = 0;
    int iters = 1000;
    int warmup = 100;
    int seq_len = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dev") == 0 && i + 1 < argc)
            dev_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
            iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc)
            seq_len = atoi(argv[++i]);
    }

    cudaSetDevice(dev_id);

    DecodeMicrokernelConfig cfg = attention_decode_config_fixed128();
    Descriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.seq_id = 1;
    desc.kv_block_offset = 0;
    desc.num_kv_blocks = 1;
    desc.output_token_offset = 0;

    uint8_t *d_kv = NULL;
    SampleState *d_sample = NULL;
    uint32_t *d_token = NULL, *d_bytes = NULL, *d_tiles = NULL;
    uint64_t *d_cycles = NULL;
    cudaMalloc(&d_kv, KV_LAYOUT_BLOCK_BYTES);
    cudaMalloc(&d_sample, sizeof(SampleState));
    cudaMalloc(&d_token, sizeof(uint32_t));
    cudaMalloc(&d_bytes, sizeof(uint32_t));
    cudaMalloc(&d_tiles, sizeof(uint32_t));
    cudaMalloc(&d_cycles, sizeof(uint64_t));
    cudaMemset(d_kv, 0, KV_LAYOUT_BLOCK_BYTES);
    cudaMemset(d_sample, 0, sizeof(SampleState));

    for (int i = 0; i < warmup; i++) {
        bench_decode_kernel<<<1, 32, cfg.shared_bytes>>>(
            d_token, d_cycles, d_bytes, d_tiles, d_kv, d_sample, desc);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        bench_decode_kernel<<<1, 32, cfg.shared_bytes>>>(
            d_token, d_cycles, d_bytes, d_tiles, d_kv, d_sample, desc);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    uint32_t token = 0, bytes = 0, tiles = 0;
    uint64_t cycles = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(&token, d_token, sizeof(token), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bytes, d_bytes, sizeof(bytes), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tiles, d_tiles, sizeof(tiles), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost);

    printf("GB300 RL Runtime — Decode Microkernel Scaffold\n");
    printf("  Mode:               v0.2.2-a scaffold\n");
    printf("  Device:             %d\n", dev_id);
    printf("  Head dim:           %u\n", cfg.head_dim);
    printf("  Seq len (reported): %d\n", seq_len);
    printf("  Tokens/block:       %u\n", cfg.tokens_per_block);
    printf("  Tile tokens:        %u\n", cfg.tile_tokens);
    printf("  Prefetch stages:    %u\n", cfg.prefetch_stages);
    printf("  Shared bytes:       %u\n", cfg.shared_bytes);
    printf("  Iterations:         %d\n", iters);
    printf("  Wall time:          %.3f ms\n", ms);
    printf("  Avg ns/iter:        %.1f\n", (ms * 1.0e6) / iters);
    printf("  Bytes touched/iter: %u\n", bytes);
    printf("  Tile count/iter:    %u\n", tiles);
    printf("  Cycle estimate:     %lu\n", (unsigned long)cycles);
    printf("  Token sample:       %u\n", token);
    printf("  Note:               shared-memory staging scaffold only; real attention math is not implemented yet.\n");

    cudaFree(d_kv);
    cudaFree(d_sample);
    cudaFree(d_token);
    cudaFree(d_bytes);
    cudaFree(d_tiles);
    cudaFree(d_cycles);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
