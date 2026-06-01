# Architecture

This document maps every runtime component to its role in the RL
inference pipeline and explains the design decisions.

## Design Goal

Build a close-to-metal C/CUDA runtime for RL inference where:

1. The **hot path** (per-token decode loop) contains zero syscalls,
   zero page faults, zero malloc/free, zero CPU scheduler wakeups,
   and zero per-token CUDA kernel launches.
2. The **control plane** (rollout state machine, pipeline queues,
   reward handoff) is lock-free and lives in hardware-visible
   memory (NVLink-C2C coherent or pinned host memory).
3. The **KV cache** is pre-allocated and arena-managed — no dynamic
   allocation after init.

## System Overview

```
 ┌─────────────────────────────────────────────────────────────┐
 │                        Host (CPU)                           │
 │                                                             │
 │  rollout_slab  ──►  RolloutPipeline  ──►  metrics/hp_guard  │
 │  (bitmap slab)      (6 ID rings)        (instrumentation)   │
 │       │                    │                    │            │
 │       ▼                    ▼                    ▼            │
 │  ┌────────┐    ┌──────────────────┐    ┌──────────────┐     │
 │  │ roll-  │    │ free  prefill    │    │ TraceRing    │     │
 │  │ out_t  │    │ decode reward    │    │ (1M entries) │     │
 │  │ state  │    │ traj   done      │    └──────────────┘     │
 │  │ CAS    │    └──────────────────┘                          │
 │  └────────┘           │                                      │
 │                       ▼                                      │
 │              ┌────────────────┐                              │
 │              │  CommandRing   │ ◄──── SPSC, CPU prod          │
 │              │  (Descriptor)  │       GPU cons               │
 │              └───────┬────────┘                              │
 │                      │   NVLink-C2C / coherent memory         │
 ├──────────────────────┼───────────────────────────────────────┤
 │                 GPU  │                                        │
 │                      ▼                                        │
 │  ┌──────────────────────────────────────────────────────┐    │
 │  │          Persistent decode_worker (1/SM)              │    │
 │  │                                                      │    │
 │  │  poll ring ─► read desc ─► cp.async KV ─► decode     │    │
 │  │                              ─► sample ─► comp_ring   │    │
 │  └──────────────────────────────────────────────────────┘    │
 │                      │                                        │
 │                      ▼                                        │
 │              ┌──────────────┐                                 │
 │              │ Completion   │ ◄──── GPU prod, CPU cons         │
 │              │ Ring         │                                  │
 │              └──────────────┘                                  │
 │                      │                                         │
 │                      ▼                                         │
 │              ┌────────────────┐                                │
 │              │  RewardRing    │ ◄──── handoff to reward worker │
 │              └────────────────┘                                │
 └─────────────────────────────────────────────────────────────┘
```

## Hot Path vs Init Path

Every operation in the runtime is classified into one of two paths:

| Path | When | What's allowed |
|------|------|----------------|
| **Init** | Once at startup | mmap, cudaMalloc, memset, NUMA bind, kernel launch |
| **Hot path** | Every decode step | Atomic loads/stores, cp.async, arithmetic, sampling |

The hot-path invariant:

> No syscall, no page fault, no malloc/free, no scheduler wakeup,
> no per-token CUDA kernel launch in the per-token decode loop.

This is enforced by:
- **Hot-path guard** (`hotpath_guard.h/c`): counts every allocation
  call and flags violations when the guard is active.
- **Static allocation**: all memory (rings, arena, rollouts) is
  pre-allocated at init. `pipeline_push/pop` uses only atomics.
- **Persistent GPU workers**: launched once at init, poll the ring
  in a loop, never return to the host between tokens.

See `docs/hotpath.md` for the full classification table.

## Pipeline Stages

A rollout moves through 6 stages, each represented by a lock-free
SPSC ring of rollout IDs:

```
 free_q  ──► prefill_q  ──► decode_q  ──► reward_q  ──► traj_q  ──► done_q
   │            │              │              │             │           │
   │            │              │              │             │           │
   ▼            ▼              ▼              ▼             ▼           ▼
 alloc      prefill       decode step     score       record      release
 rollout    KV compute    (GPU worker)    reward      trajectory  resources
```

Each ring is a fixed-size (4096 slots) power-of-2 SPSC queue with
cacheline-separated head/tail indices. Push/pop are lock-free and
use acquire/release memory ordering.

## State Machine

A rollout's `state` field is modified only through
`rollout_transition()`, which uses `__atomic_compare_exchange_n`
with ACQ_REL semantics. The valid transitions are:

```
 FREE ──► PREFILL_READY ──► DECODING ──► REWARD_PENDING ──► TRAJECTORY_READY ──► DONE ──► FREE
                                                                     │
                                                           (or back to DECODING
                                                            for multi-step reward)
```

Invalid transitions (e.g., FREE → DONE) are rejected at runtime by
`rollout_is_valid_transition()` and return -1.

## Backpressure & Flow Control

The pipeline includes credit-based flow control to prevent any
single stage from overflowing:

```c
typedef struct {
    uint32_t max_decode_credits;     // max concurrent decode rollouts
    uint32_t max_reward_credits;     // max pending reward evaluations
    uint32_t max_trajectory_credits; // max queued trajectories
    uint32_t kv_block_limit;         // max KV blocks in use
    uint32_t decode_used;            // current decode occupancy
    uint32_t reward_used;            // current reward occupancy
    uint32_t trajectory_used;        // current trajectory occupancy
    uint32_t kv_blocks_used;         // current KV block count
} PipelineCredits;
```

`pipeline_try_push()` checks credits before admitting a rollout.
If a stage's credit limit is reached, the caller must wait or
drain completions first. This prevents cascade failures when
the pipeline backs up.

## Scheduling Policies

When multiple rollouts are queued, `pipeline_schedule()` selects
the next one based on the active policy:

| Policy | Behavior | Use case |
|--------|----------|----------|
| `SCHED_FIFO` | Strict queue order | Fairness baseline |
| `SCHED_SHORTEST_REMAINING` | Pick rollout with fewest remaining tokens | Minimize p99 latency |
| `SCHED_PREFIX_SHARING` | Prefer rollout sharing prefix with recently scheduled | Maximize KV cache reuse |

## Copy-on-Write Prefix KV

RL often creates many rollouts from the same prompt. The COW KV
system avoids duplicating the shared prefix:

```
 shared prefix KV (blocks 0-3)
     │
     ├── rollout A delta (blocks 4-5)
     ├── rollout B delta (blocks 4-6)
     └── rollout C delta (blocks 4-5, 7)
```

Each prefix has a reference count. A branch holds a pointer to
the prefix plus its own delta blocks. When all branches release
their reference, the prefix is freed.

The GPU kernel `kv_branch_resolve()` computes the total KV offset
for a batch of branches in one call — no CPU round-trip.

## Tracing

The trace ring (`include/trace.h`, `src/trace.c`) records
nanosecond timestamps for 10 event types across the pipeline.
After a benchmark run, the trace report computes end-to-end
latencies for 8 paired event types with p50/p90/p99 percentiles.

This makes latency regressions visible per-commit.

## RTL Control-Plane Model

The repo also contains a SystemVerilog control-plane model under `rtl/`.
This exists so the queueing protocol can be exercised as hardware logic
without pretending to model transformer compute or GPU internals.

Relationship to the C/CUDA runtime:

- software fast path:
  `infer_submit_decode() -> hw_desc_t -> hw_ring -> mmio_write32()`
- RTL model:
  `host_desc -> desc_ring -> rollout_worker_fsm -> completion_ring`

The ownership model maps cleanly:

- software producer `tail` ownership corresponds to `push_valid/push_ready`
- software consumer `head` ownership corresponds to `pop_valid/pop_ready`
- completion backpressure in software maps to stalled `host_comp_ready`
  in RTL

Completion backpressure matters because a host-facing completion path
must never silently drop results just because downstream is not ready
for a few cycles.

The longer-term direction is C + Verilator co-simulation so the software
submit path can drive the RTL descriptor engine directly.

## Instrumentation

Three layers of instrumentation, each adding more detail:

| Layer | Component | Cost | When to use |
|-------|-----------|------|-------------|
| 1. Counters | `RuntimeMetrics` | ~1 ns/inc (relaxed atomic) | Always on |
| 2. Hot-path guard | `HotpathGuard` | ~1 ns/check | CI / debug |
| 3. Tracing | `TraceRing` | ~40 ns/push | Benchmarks only |

`make bench` uses layer 1. `make bench-pipeline` uses layers 1+2.
`make bench-trace` uses all three.

## File Map

```
include/
  descriptor.h      28-byte decode-step descriptor
  ring.h            Lock-free SPSC command ring (CPU→GPU)
  completion.h      Completion ring (GPU→CPU)
  arena.h           Hugepage-backed KV block arena
  prefetch.h        cp.async software-pipelined KV loader
  sample.h          GPU-resident token sampling
  numa.h            NUMA-local hugepage allocation
  rollout.h         Rollout state machine + slab allocator
  pipeline.h        Multi-queue pipeline + backpressure + scheduling
  metrics.h         Cacheline-padded hot-path counters
  hotpath_guard.h   Hot-path allocation violation guard
  trace.h           Ring-buffer tracing (10 event types)
  reward.h          SPSC reward descriptor ring
  kv_prefix.h       Copy-on-write prefix KV
  hw_desc.h         64-byte hardware-facing descriptor
  hw_ring.h         Cacheline-owned hardware command/done rings
  infer_submit.h    Host submit API for the hardware fast path
  mmio.h            MMIO-style doorbell helpers

src/
  ring.c            Ring allocation in coherent memory
  arena.c           Bitmap-based O(1) arena allocator
  numa.c            mbind-based NUMA binding
  main.c            Host runtime (init, dispatch, poll)
  rollout.c         Slab allocator + CAS transitions
  pipeline.c        ID ring push/pop + credts + scheduling
  metrics.c         Metrics init + formatted output
  hotpath_guard.c   Allocation tracking + violation stderr
  trace.c           Trace push + pair-latency report
  reward.c          Reward ring + mock scoring
  kv_prefix.c       Prefix/branch register + refcount
  hw_ring.c         Hardware-facing ring allocation and atomics
  infer_submit.c    Build descriptor and ring MMIO-style doorbell
  mmio.c            Translation-unit anchor for MMIO helpers

cu/
  worker.cu         Persistent GPU decode worker (1 warp/SM)
  prefetch.cu       cp.async pipeline device code
  sample.cu         xoshiro256** + top-k/p/temperature
  kv_prefix.cu      GPU branch offset resolve kernel
  reward.cu         GPU reward scoring kernel

bench/
  bench_pipeline.cu        Full RL pipeline benchmark
  bench_trace_pipeline.cu  Pipeline benchmark with tracing
  bench_cow_prefix.cu      COW prefix KV memory savings benchmark
  bench_control_tax.cu     Control-plane tax comparison benchmark
  bench_hw_fastpath.c      Hardware descriptor / doorbell batching benchmark

test/
  test_bench.cu     11 unit tests + GPU pipeline test
  test_hw_ring.c    Hardware fastpath tests

lab/
  01_false_sharing  Cache line contention (C, pthreads)
  02_spsc_ring      SPSC ring from scratch (C, atomics)
  03_hugepage_tlb   4K vs 2M page TLB comparison (C, mmap)
  04_syscall_vs_poll eventfd vs polling cost (C, eventfd)
  05_doorbell_mock  Producer/consumer doorbell (C, atomics)
  06_memory_ordering Publication ordering lab

rtl/
  desc_pkg.sv       Packed descriptor and completion definitions
  mmio_regs.sv      MMIO-style doorbell register model
  desc_ring.sv      Descriptor ready/valid ring
  completion_ring.sv Completion ready/valid ring
  rollout_worker_fsm.sv Fake decode / control-plane worker
  rl_runtime_top.sv Top-level RTL composition
  tb_rl_runtime_top.sv Basic RTL testbench
```
