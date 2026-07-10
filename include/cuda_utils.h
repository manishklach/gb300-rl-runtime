#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err), (int)_err); \
        goto cleanup; \
    } \
} while (0)
