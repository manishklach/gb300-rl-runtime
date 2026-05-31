#include "kv_layout.h"
#include <stdio.h>

int
main(void)
{
    KvLayoutDesc desc = kv_layout_desc_fixed128();
    printf("GB300 RL Runtime — KV Layout Scaffold\n");
    printf("  Head dim:           %u\n", desc.head_dim);
    printf("  Tokens/block:       %u\n", desc.token_count);
    printf("  Scalar bytes:       %u\n", desc.scalar_bytes);
    printf("  Vector bytes:       %u\n", desc.vec_bytes);
    printf("  K stride bytes:     %u\n", desc.k_stride_bytes);
    printf("  V stride bytes:     %u\n", desc.v_stride_bytes);
    printf("  K bytes/block:      %u\n", KV_LAYOUT_K_BYTES);
    printf("  V bytes/block:      %u\n", KV_LAYOUT_V_BYTES);
    printf("  Total block bytes:  %u\n", desc.block_bytes);
    printf("  Alignment:          %u\n", desc.alignment);
    printf("  V plane base:       %u\n", kv_layout_v_plane_base_bytes());
    printf("  Example block[3]:   %u\n", kv_layout_block_offset_bytes(3));
    printf("  Example K token[7]: %u\n", kv_layout_k_offset_bytes(7));
    printf("  Example V token[7]: %u\n", kv_layout_v_offset_bytes(7));
    printf("  Note:               layout scaffold only; use this to validate block math before real kernels land.\n");
    return 0;
}
