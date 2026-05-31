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
                    float *out_vec,
                    const __half *q_vec,
                    const uint8_t *kv_block,
                    SampleState *sample_st,
                    Descriptor desc,
                    uint32_t seq_len)
{
    extern __shared__ uint8_t smem[];
    if (threadIdx.x == 0) {
        DecodeStepArgs args;
        args.q_ptr = q_vec;
        args.o_ptr = out_vec;
        args.seq_len = seq_len;
        args.head_dim = DECODE_FIXED_HEAD_DIM;
        args.kv_block_base_idx = desc.kv_block_offset;
        args.kv_block_count = 1;
        args.output_token_offset = desc.output_token_offset;
        DecodeStepResult result =
            attention_decode_step_fixed128(&desc, &args, kv_block, sample_st, smem);
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
    int seq_len = 32;

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
    __half *d_q = NULL;
    float *d_out = NULL;
    uint32_t *d_token = NULL, *d_bytes = NULL, *d_tiles = NULL;
    uint64_t *d_cycles = NULL;
    __half h_q[DECODE_FIXED_HEAD_DIM];
    __half *h_kv = (__half *)malloc(KV_LAYOUT_BLOCK_BYTES);
    float h_out[DECODE_FIXED_HEAD_DIM];
    double out_sum = 0.0;

    if (seq_len < 1)
        seq_len = 1;
    if (seq_len > (int)KV_LAYOUT_TOKENS_PER_BLOCK)
        seq_len = (int)KV_LAYOUT_TOKENS_PER_BLOCK;

    for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++)
        h_q[d] = __float2half_rn((float)((int)(d % 17U) - 8) / 8.0f);
    for (uint32_t t = 0; t < KV_LAYOUT_TOKENS_PER_BLOCK; t++) {
        for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++) {
            const float k_val = (float)(((int)((t + 1U) * ((d % 13U) + 1U)) % 23) - 11) / 32.0f;
            const float v_val = (float)(((int)((t + 3U) * ((d % 11U) + 5U)) % 29) - 14) / 29.0f;
            h_kv[t * DECODE_FIXED_HEAD_DIM + d] = __float2half_rn(k_val);
            h_kv[(KV_LAYOUT_K_BYTES / sizeof(__half)) + t * DECODE_FIXED_HEAD_DIM + d] =
                __float2half_rn(v_val);
        }
    }

    cudaMalloc(&d_kv, KV_LAYOUT_BLOCK_BYTES);
    cudaMalloc(&d_sample, sizeof(SampleState));
    cudaMalloc(&d_q, sizeof(h_q));
    cudaMalloc(&d_out, sizeof(h_out));
    cudaMalloc(&d_token, sizeof(uint32_t));
    cudaMalloc(&d_bytes, sizeof(uint32_t));
    cudaMalloc(&d_tiles, sizeof(uint32_t));
    cudaMalloc(&d_cycles, sizeof(uint64_t));
    cudaMemcpy(d_kv, h_kv, KV_LAYOUT_BLOCK_BYTES, cudaMemcpyHostToDevice);
    cudaMemset(d_sample, 0, sizeof(SampleState));
    cudaMemcpy(d_q, h_q, sizeof(h_q), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(h_out));

    for (int i = 0; i < warmup; i++) {
        bench_decode_kernel<<<1, 32, cfg.shared_bytes>>>(
            d_token, d_cycles, d_bytes, d_tiles, d_out, d_q, d_kv, d_sample, desc,
            (uint32_t)seq_len);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        bench_decode_kernel<<<1, 32, cfg.shared_bytes>>>(
            d_token, d_cycles, d_bytes, d_tiles, d_out, d_q, d_kv, d_sample, desc,
            (uint32_t)seq_len);
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
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    for (uint32_t d = 0; d < DECODE_FIXED_HEAD_DIM; d++)
        out_sum += h_out[d];

    printf("GB300 RL Runtime — Fixed128 Decode Microkernel\n");
    printf("  Mode:               v0.2.2-b real math path\n");
    printf("  Device:             %d\n", dev_id);
    printf("  Head dim:           %u\n", cfg.head_dim);
    printf("  Seq len:            %d\n", seq_len);
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
    printf("  Tokens/s:           %.0f\n", (double)iters / (ms / 1000.0));
    printf("  Output checksum:    %.6f\n", out_sum);
    printf("  Output argmax:      %u\n", token);
    printf("  Note:               real fixed128 QK/softmax/V math; runtime worker still uses a synthetic query fallback when no explicit query is attached.\n");

    free(h_kv);
    cudaFree(d_kv);
    cudaFree(d_sample);
    cudaFree(d_q);
    cudaFree(d_out);
    cudaFree(d_token);
    cudaFree(d_bytes);
    cudaFree(d_tiles);
    cudaFree(d_cycles);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
