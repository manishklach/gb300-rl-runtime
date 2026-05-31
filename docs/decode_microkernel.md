# Decode Microkernel Scaffold

This document describes the `v0.2.2-a` scaffold for the first real
hardware-close decode path.

## Current State

The repository now contains the structural pieces for a fixed-shape
decode microkernel:

- `include/kv_layout.h`
- `include/attention_decode.h`
- `cu/attention_decode.cu`
- `bench/bench_decode_microkernel.cu`
- `bench/bench_kv_layout.cu`

The implementation is intentionally still a scaffold:

- shared-memory staging is real
- the worker path calls the decode helper
- the decode helper reports deterministic metadata
- actual QK/softmax/V attention math is still not implemented

## Fixed Path

The first path is intentionally narrow:

- head dimension: `128`
- tokens per KV block: `32`
- scalar width: `2` bytes
- vector load target: `16` bytes
- prefetch stages: `3`

This is enough to start measuring memory layout and worker integration
without pretending the math path is done.

## Why This Split Exists

The repository now distinguishes:

- control-plane benchmarks
- decode-microkernel scaffolds
- future real attention benchmarks

That separation matters because queue correctness and hardware math are
different problems with different bottlenecks.

## What Comes Next

The next decode-focused step should replace the scaffold behavior in
`cu/attention_decode.cu` with:

1. Q load
2. KV tile staging
3. QK dot-product tiles
4. stable softmax update
5. V accumulation
6. output writeback

At that point the decode benchmark can become the first real math-path
benchmark in the repo.
