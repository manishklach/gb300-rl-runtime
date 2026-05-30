# GB300 RL Inference Runtime

A close-to-metal C/CUDA reference runtime for reinforcement learning
inference at GB300 NVL72 scale.  No page faults, no `malloc`/`free`,
no **per-token** CUDA kernel launches, no CPU scheduler wakeups in the
per-token hot path.  Persistent GPU workers are launched once at init.

## Portable, not GB300-only

The code targets GB300 because that's the interesting scale, but it
runs on **any GPU with compute capability 8.0+** (Ampere or newer).
The test bench validates the full pipeline on a single GPU with no
special hardware.

What you'd change for a non-GB300 system:

| GB300 assumption | Portable alternative |
|---|---|
| NVLink-C2C coherent command ring | `cudaHostAlloc` + `cudaHostGetDevicePointer` |
| Grace CPU NUMA topology | Drop the `mbind` calls or set node = 0 |
| Grace ARM + NVSwitch | Works on any x86 + any NVIDIA GPU |
| 72 SMs | Auto-detected from `cudaDeviceProp` |

Everything else — atomics, hugepages, `cp.async`, persistent workers,
on-device sampling — is standard CUDA C that works on any Linux system
with a modern GPU.

## Architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │                     Host (CPU Control Plane)                  │
 │                                                               │
 │  rollout_slab ──► RolloutPipeline ──► metrics/hp_guard/trace  │
 │  (bitmap slab)    (6 ID rings + credits + scheduling)         │
 │       │                    │                                   │
 │       ▼                    ▼                                   │
 │  ┌──────────┐   ┌─────────────────────┐                       │
 │  │ rollout  │   │ free   prefill      │                       │
 │  │ state    │   │ decode reward       │                       │
 │  │ machine  │   │ traj   done         │                       │
 │  │ (CAS)    │   └──────────┬──────────┘                       │
 │  └──────────┘              │                                   │
 │                            ▼                                   │
 │                   ┌────────────────┐                           │
 │                   │  CommandRing   │ ◄── CPU prod, GPU cons    │
 │                   │  (Descriptor)  │                           │
 │                   └───────┬────────┘                           │
 │                           │  coherent / pinned memory          │
 ├───────────────────────────┼───────────────────────────────────┤
 │                      GPU  │                                    │
 │                           ▼                                    │
 │  ┌───────────────────────────────────────────────────────┐     │
 │  │           Persistent decode_worker (1 warp / SM)       │     │
 │  │  poll ring ─► read desc ─► cp.async KV ─► decode      │     │
 │  │                              ─► sample ─► comp_ring    │     │
 │  └───────────────────────────────────────────────────────┘     │
 │                           │                                     │
 │                           ▼                                     │
 │                   ┌──────────────┐                              │
 │                   │ Completion   │ ◄── GPU prod, CPU cons       │
 │                   │ Ring         │                              │
 │                   └──────────────┘                              │
 │                           │                                     │
 │                    ┌───────┴────────┐                            │
 │                    │                │                            │
 │              ┌─────▼─────┐   ┌──────▼──────┐                    │
 │              │ RewardRing│   │ KV Prefix   │                    │
 │              │ (SPSC)    │   │ Table (COW) │                    │
 │              └───────────┘   └─────────────┘                    │
 └──────────────────────────────────────────────────────────────┘
```

### Hot-Path Anatomy

```
  CPU (producer)                          GPU persistent worker
  ═══════════════                          ══════════════════════
  ring_acquire()                           ┌─ poll ring tail
       │                                   │   (acquire-load)
       ▼                                   ▼
  write descriptor ──store──▶  ring  ──load──▶  read descriptor
       │                  slot  │                 │
       ▼                       │                 ▼
  ring_commit()                │            read KV arena
  (release-store head)         │            (hugepage, no TLB miss)
       │                       │                 │
       ▼                       ▼                 ▼
  ┌─────────────────────────────────────┐   decode attention
  │  key invariant:                     │   (cp.async prefetch)
  │  no syscall, no page fault,         │        │
  │  no malloc/free, no scheduler       │        ▼
  │  wakeup in the entire hot path      │   sample token
  └─────────────────────────────────────┘   (GPU-resident)
                                                  │
                                                  ▼
  CPU polls completion ◀─── store ──────────  comp_ring_push()
  (acquire-load tail)                         (release-store head)
```

## Components

| Component | File | Description |
|---|---|---|
| Work Descriptor | `include/descriptor.h` | 24-byte packed decode-step command |
| SPSC Ring | `include/ring.h`, `src/ring.c` | Lock-free producer-consumer ring in coherent memory |
| Completion Ring | `include/completion.h` | GPU→CPU result notification (mirror of command ring) |
| KV Arena | `include/arena.h`, `src/arena.c` | Hugepage-backed slab allocator with O(1) acquire/release |
| Prefetch Pipeline | `include/prefetch.h`, `cu/prefetch.cu` | `cp.async` software-pipelined KV block loader |
| Sampling | `include/sample.h`, `cu/sample.cu` | GPU-resident top-k / top-p / temperature sampling |
| Persistent Worker | `cu/worker.cu` | GPU SM decode loop — polls ring, loads KV, runs attention |
| NUMA Helpers | `include/numa.h`, `src/numa.c` | `mbind`-based NUMA-local hugepage allocation |
| Host Runtime | `src/main.c` | Init, dispatch loop, completion polling |
| Rollout State Machine | `include/rollout.h`, `src/rollout.c` | CAS-based rollout lifecycle with valid transition table |
| Rollout Pipeline | `include/pipeline.h`, `src/pipeline.c` | 6-queue RL pipeline (free/prefill/decode/reward/trajectory/done) |
| Runtime Metrics | `include/metrics.h`, `src/metrics.c` | Cacheline-padded hot-path counters with formatted output |
| Hot-Path Guard | `include/hotpath_guard.h`, `src/hotpath_guard.c` | Detects malloc/cudaMalloc/page faults in hot path |
| Copy-on-Write Prefix KV | `include/kv_prefix.h`, `src/kv_prefix.c`, `cu/kv_prefix.cu` | Shared prefix KV with per-rollout delta branches |
| Reward Pipeline | `include/reward.h`, `src/reward.c`, `cu/reward.cu` | Mock reward/verifier ring with GPU scoring kernel |
| Pipeline Benchmark | `bench/bench_pipeline.cu` | Full RL pipeline benchmark with hot-path guard verification |
| Trace Pipeline Benchmark | `bench/bench_trace_pipeline.cu` | Pipeline benchmark with nanosecond tracing and latency percentiles |
| Tracing | `include/trace.h`, `src/trace.c` | Ring-buffer trace entries with pair-latency report (p50/p90/p99) |
| Scheduling Policies | `include/pipeline.h`, `src/pipeline.c` | FIFO / shortest-remaining / prefix-sharing scheduling |
| Pipeline Backpressure | `include/pipeline.h`, `src/pipeline.c` | Credit-based flow control per pipeline stage |
| COW Prefix KV Benchmark | `bench/bench_cow_prefix.cu` | Memory savings comparison: COW vs full-duplicate KV |
| Control-Plane Tax Benchmark | `bench/bench_control_tax.cu` | Syscall vs polling vs persistent worker comparison |

## What This Is Not

This is not a replacement for vLLM, TensorRT-LLM, SGLang, or JAX.

This is a **reference fast-path** showing how an RL inference runtime
could structure fixed KV ownership, CPU→GPU command rings, persistent
decode workers, hugepage-backed memory, cacheline-aware queues, and
async reward handoff.

The goal is to study the control-plane mechanics, not to outperform
production inference stacks today.

## Documentation

| File | What it covers |
|---|---|
| `docs/architecture.md` | Full architecture map, hot path vs init path, component interactions |
| `docs/hotpath.md` | Every operation classified as init vs hot path |
| `docs/metrics.md` | Target metrics and benchmark commands (all benchmarks) |
| `docs/tracing.md` | Trace event types, latency pairs, example output |

## Build

Requires CUDA 12.x+ and `libnuma-dev`.

```bash
make               # build library + test bench + all benchmarks
make test          # run unit tests (including rollout, pipeline, metrics, guard, reward, prefix KV)
make bench         # benchmark: 1M tokens through ring+worker
make bench-pipeline # benchmark: full RL pipeline with rollouts, state machine, reward, hot-path guards
make bench-trace   # benchmark with nanosecond tracing + latency percentiles
make bench-cow     # COW prefix KV memory savings benchmark
make bench-tax     # control-plane tax comparison (syscall vs polling vs persistent worker)
make bench-all     # run all benchmarks
```

## Benchmark Snapshot

```
Hardware:
  GPU:      H100 (or your GPU here)
  CPU:      x86-64 / Grace ARM
  OS:       Linux
  CUDA:     12.x
  Build:    make bench-all

Results (run `make bench-all` on your hardware):
  Ring throughput:                > 50 M ops/s
  Full pipeline tokens/s:         > 1M tokens/s
  Pipeline rollouts/s:            > 10K rollouts/s
  COW prefix KV memory saved:     > 90% for 10K branches
  Control-plane tax:
    syscall per step:             ~X ns (baseline)
    userspace polling:            ~Y ns (fast, CPU-hungry)
    persistent worker (this):     ~Z ns (fastest, no CPU tax)
  Hot-path mallocs:               0 (clean)
  Post-init page faults:          0
  Per-token kernel launches:      0
```

Run on your hardware and open a PR to update this table.

## Labs

The `lab/` directory contains five self-contained C experiments
that teach the close-to-metal concepts used in the runtime:

| Lab | What it teaches | Runs on |
|---|---|---|
| `01_false_sharing` | Cache line contention — MESI protocol, padding | any Linux |
| `02_spsc_ring` | Lock-free ring buffer from scratch — atomics, memory ordering | any Linux |
| `03_hugepage_tlb` | 4K vs 2M page TLB miss comparison — why hugepages matter | Linux w/ hugepages |
| `04_syscall_vs_poll` | eventfd wakeup vs shared-memory polling — syscall cost | any Linux |
| `05_doorbell_mock` | Producer/consumer with doorbell — device queue model | any Linux |

Each lab is standalone — `cd lab/01_false_sharing && make run`.

```bash
make labs      # build all labs
make lab-run   # run all labs sequentially
```

## Design Rules

1. **Pre-fault everything** — no runtime page faults
2. **No cudaMalloc after init** — static slab allocation
3. **CPU stays out of the data path** — descriptors only
4. **NUMA-local memory** — `mbind(MPOL_BIND)` on every allocation
5. **Reward is GPU-resident** — no PCIe round-trips for scoring
6. **NVLink-C2C for coordination** — coherent rings, no DMA
7. **Rollouts are hardware-visible state machines** — CAS transitions through 6 pipeline stages, no Python/CPU per-token control
8. **Hot-path guards verify zero-alloc** — `hotpath_guard` catches accidental `malloc`/`cudaMalloc` in per-token loops
9. **Copy-on-write prefix KV** — shared prompt KV across rollouts, only per-rollout deltas allocated
10. **Credit-based backpressure** — every pipeline stage has a max occupancy; `pipeline_try_push` blocks well before ring-full
11. **Multiple scheduling policies** — FIFO, shortest-remaining-first, and prefix-sharing policies for the decode queue

## License

MIT
