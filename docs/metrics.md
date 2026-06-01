# Metrics & Benchmarks

Target metrics for each component and commands to measure them.

## Ring Throughput

| Metric | Target | How to measure |
|--------|--------|----------------|
| Enqueue latency (single-thread) | < 20 ns | `lab/02_spsc_ring/bench` |
| Dequeue latency (single-thread) | < 20 ns | (same bench — symmetric) |
| Producer-consumer throughput    | > 50 M ops/s | `lab/02_spsc_ring/bench 50000000` |
| False-sharing overhead          | < 5× penalty | `lab/01_false_sharing/bench` |

```bash
cd lab/02_spsc_ring && make run
```

## TLB & Hugepages

| Metric | 4K pages | 2M pages | How to measure |
|--------|:--------:|:--------:|----------------|
| Sequential scan (1 GB)  | baseline | ~2-5× faster | `lab/03_hugepage_tlb/bench` |
| Random access (1 GB)    | baseline | ~5-20× faster | `lab/03_hugepage_tlb/bench` |
| dTLB misses (seq scan)  | 200K+    | < 1K     | `make perf` in lab 3 |

```bash
cd lab/03_hugepage_tlb && make perf
```

## Syscall Cost

| Mechanism | Per-op latency | How to measure |
|-----------|:-------------:|----------------|
| `eventfd` write+read     | ~1-5 µs | `lab/04_syscall_vs_poll/bench` |
| Busy-poll (shared mem)   | ~20-50 ns | (same bench) |
| `sched_yield` + poll     | ~200-500 ns | (same bench) |

```bash
cd lab/04_syscall_vs_poll && make run
```

## Memory Ordering

| Metric | What to watch | How to measure |
|--------|----------------|----------------|
| Stale descriptor observations | Non-zero in broken publish mode | `lab/06_memory_ordering/bench` |
| Visibility gap | Window between publishing the flag and payload visibility | (same bench) |
| Correct publish behavior | Zero stale reads in release/acquire mode | (same bench) |

```bash
cd lab/06_memory_ordering && make run
```

## Doorbell Queue (Mock)

| Metric | Target | How to measure |
|--------|--------|----------------|
| Enqueue + doorbell + dequeue | < 50 ns | `lab/05_doorbell_mock/bench` |

```bash
cd lab/05_doorbell_mock && make run
```

## Full Pipeline (Single GPU)

Measured by `test/test_bench.cu --bench`:

| Metric | H100 target | Notes |
|--------|:-----------:|-------|
| Tokens/s (mock decode)  | > 1M / s | Ring + worker + completion loop |
| Worker poll latency      | < 100 ns | Time from CPU enqueue to GPU consume |
| Completion notification  | < 200 ns | GPU write → CPU poll visible |

```bash
make bench                                    # default: 100K tokens
make bench ARGS="--bench 1000000"             # 1M tokens
```

## Pipeline Benchmark

Measured by `bench/bench_pipeline.cu` (full RL flow simulation):

| Metric | Target | How to measure |
|--------|:------:|----------------|
| Rollout throughput       | > 10K rollouts/s | `make bench-pipeline ARGS="-r 10000 -t 128"` |
| Descriptor posting rate  | > 1M desc/s      | Included in bench output |
| Reward handoffs          | matched to decode | Included in bench output |
| Hot-path mallocs         | 0 absolute        | Guard counter in output |
| Hot-path cudaMallocs     | 0 absolute        | Guard counter in output |
| Descriptor post → commit | < 100 ns p50     | `make bench-trace` latency report |
| Desc post → GPU dequeue  | < 5 µs p50       | `make bench-trace` latency report |

```bash
make bench-pipeline                           # default: 1000 rollouts, 128 tokens each
make bench-pipeline ARGS="-r 10000 -t 256"   # 10K rollouts, 256 tokens each
make bench-all                                # runs both benchmarks
```

## Trace Pipeline Benchmark

Measured by `bench/bench_trace_pipeline.cu` with ring-buffer tracing:

| Metric | Source |
|--------|--------|
| Descriptor post → commit latency | Trace pair (p50/p90/p99) |
| Desc post → GPU dequeue latency | Trace pair (p50/p90/p99) |
| Desc post → completion latency | Trace pair (p50/p90/p99) |
| GPU dequeue → completion latency | Trace pair (p50/p90/p99) |
| Rollout lifetime | Trace pair (p50/p90/p99) |
| Reward → trajectory handoff | Trace pair (p50/p90/p99) |

```bash
make bench-trace                              # default: 1000 rollouts, 128 tokens each
make bench-trace ARGS="-r 10000 -t 256"      # 10K rollouts with tracing
```

## COW Prefix KV Benchmark

Measured by `bench/bench_cow_prefix.cu`:

| Metric | 10K branches target | How to measure |
|--------|:-------------------:|----------------|
| Memory saved vs full duplicate | > 90% | `make bench-cow` |
| Prefix refcount after branches | == n_branches | Included in output |
| Branch alloc time | < 100 ns/op | Included in output |
| Branch resolve time | < 50 ns/op | Included in output |

```bash
make bench-cow                               # default: 10K branches
make bench-cow ARGS="100000"                 # 100K branches
```

## Control-Plane Tax Benchmark

Measured by `bench/bench_control_tax.cu`:

| Mode | Mechanism | Expected latency |
|------|-----------|:----------------:|
| A | eventfd write+read per step | ~1-5 µs |
| B | Userspace polling (atomic spin) | ~20-50 ns |
| C | Persistent worker + ring (this runtime) | ~10-30 ns |

```bash
make bench-tax                               # default: 1M iterations
make bench-tax ARGS="10000000"               # 10M iterations
```

## Hardware Fastpath Benchmark

Measured by `bench/bench_hw_fastpath.c`:

| Metric | Target | How to measure |
|--------|:------:|----------------|
| Descriptor submit throughput | measure on your CPU | `make bench-hw-fastpath` |
| Worker-sim completion throughput | measure on your CPU | `make bench-hw-fastpath` |
| Doorbell batching tradeoff | compare batch 1 / 8 / 32 / 64 | Included in output |
| Submit latency p50/p99 | lower is better | Included in output |

```bash
make bench-hw-fastpath
make bench-hw-fastpath ARGS="500000"
```

## GPU-Resident Scheduler Benchmark

Measured by `bench/bench_gpu_scheduler.cu`:

| Metric | Target | How to measure |
|--------|:------:|----------------|
| Rollouts/s (GPU-managed)   | > 20K rollouts/s | `make bench-gpu-scheduler` |
| Tokens/s (GPU-managed)     | > 2M tokens/s    | Included in output |
| CPU dispatches per rollout | 1 (request only)  | Architecture property |
| CPU polls per rollout      | 1 (done only)     | Architecture property |

```bash
make bench-gpu-scheduler                      # default: 10K rollouts, 128 tokens
make bench-gpu-scheduler ARGS="-r 50000 -t 256"
```

## Prefetch Microbenchmark

Measured by `bench/bench_prefetch.cu`:

| Metric | Target | How to measure |
|--------|:------:|----------------|
| Baseline shared-copy bandwidth | measure on your GPU | `make bench-prefetch` |
| `cp.async` staged bandwidth | measure on your GPU | `make bench-prefetch` |
| `cp.async` speedup | > 1.0x on Ampere+ | Included in output |

```bash
make bench-prefetch
make bench-prefetch ARGS="8192"
```

## What Each Benchmark Proves

| Benchmark | Proves |
|-----------|--------|
| `bench` | Ring + GPU worker baseline throughput |
| `bench-pipeline` | End-to-end RL rollout flow with hot-path guard |
| `bench-trace` | Nanosecond latency breakdown (p50/p90/p99) |
| `bench-cow` | Memory saved by shared prefix KV |
| `bench-tax` | Control-plane overhead: syscall vs polling vs persistent worker |
| `bench-hw-fastpath` | Hardware-facing descriptor rings, MMIO-style doorbells, and batching tradeoffs |
| `bench-gpu-scheduler` | GPU-managed rollout lifecycle — zero CPU per-token work |
| `bench-prefetch` | Isolated KV staging bandwidth with and without `cp.async` |

## GPU-Free CUDA Validation

These are compile-only checks, not runtime benchmarks.

| Target | What it validates |
|--------|-------------------|
| `make cuda-compile-check` | `nvcc` can compile the runtime, tests, and benchmark translation units to objects |
| `make cuda-ptx-check` | `nvcc` can lower key CUDA translation units to PTX for inspection |

## How to Contribute a Benchmark

1. Add a new `.cu` or `.c` file under `test/` or `lab/`.
2. Run with `perf stat` where applicable.
3. Open a PR with before/after numbers.
