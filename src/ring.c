#include "ring.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* Allocate a CommandRing in NUMA-local, page-aligned memory.
 * For the prototype we use mmap; in production this would use
 * cudaHostAlloc with cudaHostAllocMapped or NVLink-C2C coherent
 * allocator. */
CommandRing *
ring_create(void) {
  void *p = mmap(NULL, sizeof(CommandRing),
                 PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                 -1, 0);
  if (p == MAP_FAILED) {
    /* fall back to 4K pages if hugepages not configured */
    p = mmap(NULL, sizeof(CommandRing),
             PROT_READ | PROT_WRITE,
             MAP_ANONYMOUS | MAP_PRIVATE,
             -1, 0);
    if (p == MAP_FAILED)
      return NULL;
  }
  memset(p, 0, sizeof(CommandRing));
  return (CommandRing *)p;
}

void
ring_destroy(CommandRing *ring) {
  munmap(ring, sizeof(CommandRing));
}
