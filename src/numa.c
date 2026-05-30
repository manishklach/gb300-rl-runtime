#include "numa.h"
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>

void *
numa_alloc_hugepages(int node, size_t size) {
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                 -1, 0);
  if (p == MAP_FAILED) {
    perror("numa_alloc_hugepages: mmap");
    return NULL;
  }

  struct bitmask *mask = numa_allocate_cpumask();
  if (!mask)
    goto fail;
  numa_bitmask_setbit(mask, node);

  if (mbind(p, size, MPOL_BIND, mask->maskp, mask->size,
            MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    perror("numa_alloc_hugepages: mbind");
    numa_free_cpumask(mask);
    goto fail;
  }
  numa_free_cpumask(mask);

  /* fault every page on the target node */
  memset(p, 0, size);
  return p;

fail:
  munmap(p, size);
  return NULL;
}

void
numa_free_hugepages(void *p, size_t size) {
  if (p)
    munmap(p, size);
}
