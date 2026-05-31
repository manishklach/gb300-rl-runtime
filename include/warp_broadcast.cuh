#pragma once

#include <stdint.h>

__device__ static inline uint32_t
warp_broadcast_u32(uint32_t value, int src_lane)
{
    return __shfl_sync(0xFFFFFFFFu, value, src_lane);
}

__device__ static inline uint64_t
warp_broadcast_u64(uint64_t value, int src_lane)
{
    union {
        uint64_t u64;
        uint2    u32x2;
    } parts;

    parts.u64 = value;
    parts.u32x2.x = __shfl_sync(0xFFFFFFFFu, parts.u32x2.x, src_lane);
    parts.u32x2.y = __shfl_sync(0xFFFFFFFFu, parts.u32x2.y, src_lane);
    return parts.u64;
}
