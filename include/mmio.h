#pragma once

#include <stdint.h>

static inline void
mmio_wmb(void)
{
#if defined(__x86_64__)
    __asm__ volatile("sfence" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("dsb st" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static inline void
mmio_write32(volatile uint32_t *addr, uint32_t value)
{
    mmio_wmb();
    *addr = value;
}

static inline uint32_t
mmio_read32(volatile uint32_t *addr)
{
    const uint32_t value = *addr;
#if defined(__aarch64__)
    __asm__ volatile("dmb oshld" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
    return value;
}
