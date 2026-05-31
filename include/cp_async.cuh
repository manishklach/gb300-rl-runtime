#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
__device__ static inline uint32_t
cp_async_shared_addr(const void *ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return (uint32_t)__cvta_generic_to_shared(ptr);
#else
    return 0U;
#endif
}

__device__ static inline void
cp_async_ca_16(void *smem_dst, const void *gmem_src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t smem_addr = cp_async_shared_addr(smem_dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_src)
        : "memory");
#else
    const uint8_t *src = (const uint8_t *)gmem_src;
    uint8_t *dst = (uint8_t *)smem_dst;
    for (int i = 0; i < 16; i++)
        dst[i] = src[i];
#endif
}

__device__ static inline void
cp_async_commit(void)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ static inline void
cp_async_wait_all(void)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#else
    __syncthreads();
#endif
}
#endif
