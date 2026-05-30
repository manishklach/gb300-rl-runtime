#pragma once
#include <stdint.h>
#include <stddef.h>

#define RING_SIZE            4096
#define MAX_KV_BLOCKS_PER_STEP 64
#define DESCRIPTOR_ALIGN     128

typedef struct __attribute__((packed)) {
  uint64_t  seq_id;              /* global trajectory ID */
  uint32_t  kv_block_offset;     /* slab index in KV arena */
  uint16_t  num_kv_blocks;       /* contiguous blocks for this step */
  uint8_t   attention_flags;     /* causal mask, sliding window, etc. */
  uint8_t   pad;
  uint32_t  output_token_offset; /* slot in completion ring */
  uint64_t  reward_cookie;       /* async reward completion tag */
} Descriptor;

_Static_assert(sizeof(Descriptor) == 24, "Descriptor must be 24 bytes");
