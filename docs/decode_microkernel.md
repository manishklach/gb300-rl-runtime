# Decode Microkernel

This document describes the `v0.2.2-b` fixed-shape path for the first real
hardware-close decode path.

## Current State

The repository now contains the structural pieces for a fixed-shape
decode microkernel:

- `include/kv_layout.h`
- `include/attention_decode.h`
- `cu/attention_decode.cu`
- `bench/bench_decode_microkernel.cu`
- `bench/bench_kv_layout.cu`

The implementation now has one mathematically real fixed path:

- shared-memory staging is real
- fixed128 decode executes real QK / softmax / V accumulation math
- the fixed128 kernel is now warp-cooperative instead of almost entirely
  lane-0 driven
- the QK score pass now uses 8-lane groups per token row rather than
  assigning an entire row to one lane
- the decode benchmark exercises explicit query and output buffers
- the CUDA test suite compares the kernel output against a host reference

What is still limited:

- the runtime path now carries explicit query/output buffers plus
  separate model-state and query-producer stages, but those stages still
  synthesize/update hidden state instead of consuming model-owned
  activations
- the model-state stage now uses explicit fixed-shape weights and a
  smooth residual-style update, but those weights are still deterministic
  scaffolding rather than trained parameters
- only one fixed shape is supported
- only one staged KV block is handled in the current real path
- the score pass now uses 8-lane row groups, but it is still a
  single-warp cooperative path, so further kernel parallelization work
  is still ahead

## Fixed Path

The first path is intentionally narrow:

- head dimension: `128`
- tokens per KV block: `32`
- scalar width: `2` bytes
- vector load target: `16` bytes
- prefetch stages: `3`

This is enough to measure a real math path without pretending the whole
runtime is model-complete.

## Why This Split Exists

The repository distinguishes:

- control-plane benchmarks
- fixed-shape real decode benchmarks
- future broader attention benchmarks

That separation matters because queue correctness and hardware math are
different problems with different bottlenecks.

## What Comes Next

The next decode-focused steps should broaden the real path in
`cu/attention_decode.cu`:

1. multiple KV blocks per decode step
2. replace the synthetic model-state update with real model activations
3. deeper kernel parallelism than the current single-warp cooperative execution
4. clearer split between correctness kernel and tuned kernel
The current benchmark is already the first real math-path benchmark; the
remaining work is about expanding and tuning it.
