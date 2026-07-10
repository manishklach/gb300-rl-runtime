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

_Static_assert(sizeof(CompletionRing) > sizeof(CommandRing),
               "CompletionRing must be larger than CommandRing (has extra overflow counter)");

CompletionRing *
comp_ring_create(void) {
  void *p = mmap(NULL, sizeof(CompletionRing),
                 PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                 -1, 0);
  if (p == MAP_FAILED) {
    p = mmap(NULL, sizeof(CompletionRing),
             PROT_READ | PROT_WRITE,
             MAP_ANONYMOUS | MAP_PRIVATE,
             -1, 0);
    if (p == MAP_FAILED)
      return NULL;
  }
  memset(p, 0, sizeof(CompletionRing));
  return (CompletionRing *)p;
}

void
comp_ring_destroy(CompletionRing *ring) {
  munmap(ring, sizeof(CompletionRing));
}
