#include "kv_prefix.h"
#include <cuda_runtime.h>

__device__ static int64_t
d_kv_branch_total_offset(const KVPrefixTable *t, uint32_t branch_id)
{
    if (branch_id >= MAX_BRANCHES) return -1;
    const KVBranch *b = &t->branches[branch_id];
    if (b->prefix_id >= MAX_PREFIXES)
        return b->delta_kv_offset;
    const KVPrefix *p = &t->prefixes[b->prefix_id];
    if (p->refcnt == 0)
        return b->delta_kv_offset;
    return p->kv_block_offset;
}

__global__ void
kv_branch_resolve(KVPrefixTable *t, uint32_t *branch_ids, int64_t *out_offsets, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out_offsets[i] = d_kv_branch_total_offset(t, branch_ids[i]);
}
