#include "hw_ring.h"

#include <stdlib.h>
#include <string.h>

uint32_t
hw_ring_load_acquire(volatile uint32_t *p)
{
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

void
hw_ring_store_release(volatile uint32_t *p, uint32_t value)
{
    __atomic_store_n(p, value, __ATOMIC_RELEASE);
}

hw_ring_t *
hw_ring_create(void)
{
    hw_ring_t *ring = NULL;
    if (posix_memalign((void **)&ring, 64, sizeof(*ring)) != 0)
        return NULL;

    memset(ring, 0, sizeof(*ring));
    return ring;
}

void
hw_ring_reset(hw_ring_t *ring)
{
    if (!ring)
        return;

    memset((void *)ring, 0, sizeof(*ring));
}

void
hw_ring_destroy(hw_ring_t *ring)
{
    free(ring);
}
