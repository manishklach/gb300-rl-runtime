#pragma once
#include <stddef.h>
#include <stdint.h>

/* NUMA-local memory allocation helpers.
 *
 * On GB300 each Grace CPU is its own NUMA node.  Every allocation
 * visible to a specific GPU must be bound to that GPU's nearest node. */

/* Allocate size bytes of hugepage memory on the given NUMA node.
 * Pages are pre-faulted (memset to zero).  Returns NULL on failure. */
void *numa_alloc_hugepages(int node, size_t size);

/* Free memory allocated by numa_alloc_hugepages. */
void  numa_free_hugepages(void *p, size_t size);
