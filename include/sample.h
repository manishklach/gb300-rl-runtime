#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

/* GPU-resident token sampling.
 *
 * All sampling state (PRNG, temperature, top-k mask) lives in
 * per-trajectory state allocated from the KV arena's metadata area.
 * No CPU round-trip for sampling decisions. */

#define MAX_VOCAB_SIZE 131072

/* Per-trajectory sampling state (64 bytes) */
typedef struct {
  uint64_t rng_state[4];    /* xoshiro256** state (4 x uint64) */
  float    temperature;
  uint16_t top_k;
  float    top_p;
  uint32_t vocab_size;
} SampleState;

/* Initialise sampling state for a trajectory. */
__device__ void sample_init(SampleState *st, uint64_t seed,
                            float temp, uint16_t top_k, float top_p);

/* Draw one token from logits using the configured strategy.
 * Logits are expected in device memory; the function modifies them
 * in-place (applies temperature, top-k mask, top-p filter). */
__device__ uint32_t sample_token(SampleState *st, float *logits);
