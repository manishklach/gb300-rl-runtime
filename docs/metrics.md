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

## End-to-End RL Pipeline (Future)

Once rollout state machine queues are added:

| Metric | Target |
|--------|--------|
| Trajectory init → first token | < 5 µs |
| KV block acquire              | < 50 ns |
| Decode step (mock attention)  | < 1 µs |
| Reward handoff                | < 200 ns |
| Trajectory teardown           | < 100 ns |

## How to Contribute a Benchmark

1. Add a new `.cu` or `.c` file under `test/` or `lab/`.
2. Run with `perf stat` where applicable.
3. Open a PR with before/after numbers.
