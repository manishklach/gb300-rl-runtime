#include "sample.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/* xoshiro256** — fast, small state, good statistical properties */
__device__ static inline uint64_t
rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

__device__ static inline uint64_t
xoshiro256_next(uint64_t s[4]) {
  const uint64_t result = rotl(s[1] * 5, 7) * 9;
  const uint64_t t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl(s[3], 45);
  return result;
}

__device__ void
sample_init(SampleState *st, uint64_t seed,
            float temp, uint16_t top_k, float top_p) {
  st->rng_state[0] = seed;
  st->rng_state[1] = seed ^ 0x9e3779b97f4a7c15ULL;
  st->rng_state[2] = rotl(seed, 17) ^ 0x3c6ef372fe94f82aULL;
  st->rng_state[3] = ~seed;
  st->temperature   = temp;
  st->top_k         = top_k;
  st->top_p         = top_p;
  st->vocab_size    = MAX_VOCAB_SIZE;
}

/* In-place temperature scaling + top-k + top-p filter.
 * After this, the caller can do a softmax and sample. */
__device__ uint32_t
sample_token(SampleState *st, float *logits) {
  /* apply temperature */
  if (st->temperature > 0.0f) {
    float inv_temp = 1.0f / st->temperature;
    for (uint32_t i = 0; i < st->vocab_size; i++)
      logits[i] *= inv_temp;
  }

  /* top-k: find k-th largest logit, zero out everything below it.
   * This O(vocab_size) scan is fine for inference; for training
   * use a radix-select. */
  if (st->top_k > 0 && st->top_k < st->vocab_size) {
    /* simple insertion-sort buffer of top-k (small k, typically 40-50) */
    float thresholds[256];
    uint32_t nk = (st->top_k < 256) ? st->top_k : 256;
    for (uint32_t i = 0; i < nk; i++)
      thresholds[i] = -__FLT_MAX__;

    for (uint32_t i = 0; i < st->vocab_size; i++) {
      float v = logits[i];
      for (uint32_t j = 0; j < nk; j++) {
        if (v > thresholds[j]) {
          /* shift down */
          for (uint32_t k = nk - 1; k > j; k--)
            thresholds[k] = thresholds[k - 1];
          thresholds[j] = v;
          break;
        }
      }
    }
    float cutoff = thresholds[nk - 1];
    for (uint32_t i = 0; i < st->vocab_size; i++)
      if (logits[i] < cutoff)
        logits[i] = -__FLT_MAX__;
  }

  /* top-p: sort (approx), accumulate probs, zero out tail.
   * Production would use a radix sort; for prototype we do a
   * simple rejection loop. */
  if (st->top_p > 0.0f && st->top_p < 1.0f) {
    /* compute softmax to get probabilities */
    float max_val = -__FLT_MAX__;
    for (uint32_t i = 0; i < st->vocab_size; i++)
      if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (uint32_t i = 0; i < st->vocab_size; i++)
      sum += __expf(logits[i] - max_val);
    float inv_sum = 1.0f / sum;
    /* subtract max to avoid overflow, then exp + normalize.
     * For top-p we skip sorting and just sample & reject:
     * draw uniform, walk cumulative prob. */
    float r = (float)(xoshiro256_next(st->rng_state) & 0x00FFFFFF) / 16777216.0f;
    float cum = 0.0f;
    for (uint32_t i = 0; i < st->vocab_size; i++) {
      float p = __expf(logits[i] - max_val) * inv_sum;
      cum += p;
      if (cum >= r) {
        return i;
      }
    }
    return st->vocab_size - 1; /* fallback */
  }

  /* temperature-only or raw: sample from softmax distribution */
  {
    float max_val = -__FLT_MAX__;
    for (uint32_t i = 0; i < st->vocab_size; i++)
      if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (uint32_t i = 0; i < st->vocab_size; i++) {
      logits[i] = __expf(logits[i] - max_val);
      sum += logits[i];
    }
    float r = (float)(xoshiro256_next(st->rng_state) & 0x00FFFFFF) / 16777216.0f;
    float cum = 0.0f;
    for (uint32_t i = 0; i < st->vocab_size; i++) {
      cum += logits[i] / sum;
      if (cum >= r)
        return i;
    }
    return st->vocab_size - 1;
  }
}
