# GB300 RL Inference Runtime

A close-to-metal C/CUDA reference runtime for reinforcement learning
inference at GB300 NVL72 scale.  No page faults, no `malloc`/`free`,
no **per-token** CUDA kernel launches, no CPU scheduler wakeups in the
per-token hot path.  Persistent GPU workers are launched once at init.

`gb300-rl-runtime` is a close-to-metal C/CUDA + RTL lab for RL
inference control planes.

## Why This Matters

This repo studies the control plane of RL inference: how rollout work
is described, submitted, backpressured, progressed, and completed
without per-token CPU micromanagement.

It models the path from host runtime descriptors to hardware-style
queues:

```text
C submit API
  ↓
hardware descriptor
  ↓
command ring + doorbell
  ↓
persistent worker / RTL FSM
  ↓
completion ring
  ↓
host observes completion
```

The goal is to study how RL inference work should be submitted,
progressed, backpressured, and completed without per-token CPU
micromanagement.

## Repo Stack

```text
┌──────────────────────┐
│ C/CUDA Runtime       │
│ infer_submit_decode()│
└──────────┬───────────┘
           │ hw_desc_t
           ▼
┌──────────────────────┐
│ Hardware Fast Path   │
│ ring + MMIO doorbell │
└──────────┬───────────┘
           │ descriptor contract
           ▼
┌──────────────────────┐
│ RTL Descriptor Engine│
│ desc_ring + FSM      │
└──────────┬───────────┘
           │ completion_t
           ▼
┌──────────────────────┐
│ Host Completion Path │
└──────────────────────┘
```

## v0.2.1: Correctness and Benchmark Honesty

This release is a stabilization pass focused on queue correctness,
worker shutdown safety, overflow handling, and more honest benchmark
reporting.

- SPSC command-ring accounting now uses explicit producer-tail and
  consumer-head ownership.
- Persistent workers now use `__shfl_sync` for warp value broadcast and
  reserve `__sync_warp` for synchronization only.
- Shutdown paths wait for persistent kernels to exit before freeing
  host/device resources.
- Completion and done rings now detect full conditions, count overflow
  attempts, and apply backpressure instead of silently overwriting data.
- Hot-path guard output now reports `wrapper-clean` rather than a global
  `CLEAN` claim.

## Portable, not GB300-only

The code targets GB300 because that's the interesting scale, but it
runs CUDA kernels on **any GPU with compute capability 8.0+**
(Ampere or newer).

The full host runtime is still **Linux/POSIX-oriented** today.  It uses
APIs such as `mmap`, `MAP_HUGETLB`, `mbind`, `clock_gettime`, `getopt`,
and POSIX-style filesystem/device conventions in a few tests and
benchmarks.  `make smoke` provides CPU-only queue correctness checks on
supported Linux environments without requiring a GPU.

What you'd change for a non-GB300 system:

| GB300 assumption | Portable alternative |
|---|---|---|
| NVLink-C2C coherent command ring | `cudaHostAlloc` + `cudaHostGetDevicePointer` |
| Grace CPU NUMA topology | `numa.c` now guards with `numa_available()` — skips `mbind` on non-NUMA systems |
| Grace ARM + NVSwitch | Works on any x86 + any NVIDIA GPU |
| `-arch=sm_90a` (Blackwell) | Makefile now uses multi-arch gencode: sm_80 (Ampere) through sm_90a (Blackwell) |

Everything else — atomics, hugepages, `cp.async`, persistent workers,
on-device sampling — is standard CUDA C, but the current portability
story is best described as "modern NVIDIA GPUs on Linux" rather than
"all host OSes."

## Architecture

```
 ┌────────────────────────────────────────────────────────────────┐
 │                     Host (CPU)                                 │
 │  ┌─────────────────────────────────────────────────────────┐   │
 │  │  v0.2 path (CPU per-token):                              │   │
 │  │    rollout_slab ──► Pipeline ──► CommandRing ──► GPU     │   │
 │  │    GPU ──► CompletionRing ──► CPU (one trip per token)   │   │
 │  └─────────────────────────────────────────────────────────┘   │
 │                                                                │
 │  ┌─────────────────────────────────────────────────────────┐   │
 │  │  v0.3 path (GPU-resident):                               │   │
 │  │    CPU ──► RequestRing ──► GPU (one request per rollout) │   │
 │  │    GPU ──► DoneRing ──► CPU (one done per rollout)        │   │
 │  └─────────────────────────────────────────────────────────┘   │
 │                    │                                           │
 │                    ▼                                           │
 │           ┌────────────────┐                                   │
 │           │  CommandRing   │ ──► RequestRing (v0.3)            │
 │           │  CompletionRing│ ◄── DoneRing (v0.3)               │
 │           └───────┬────────┘                                   │
 │                   │  coherent / pinned memory                   │
 ├───────────────────┼───────────────────────────────────────────┤
 │              GPU  │                                            │
 │                   ▼                                            │
 │  ┌──────────────────────────────────────────────────────┐      │
 │  │  Persistent Workers (1 warp / SM)                    │      │
 │  │                                                      │      │
 │  │  decode_worker (v0.2):                               │      │
 │  │    poll ring ─► read desc ─► decode ─► comp_ring     │      │
 │  │                                                      │      │
 │  │  rollout_worker (v0.3):                              │      │
 │  │    poll request ─► alloc slot ─► decode loop         │      │
 │  │      ─► sample ─► KV update ─► done_ring             │      │
 │  └──────────────────────────────────────────────────────┘      │
 │                   │                                            │
 │              ┌────┴────┐                                       │
 │              │         │                                       │
 │        ┌─────▼──┐  ┌───▼──────┐                               │
 │        │  KV    │  │  Reward  │                               │
 │        │ Arena  │  │  Ring    │                               │
 │        └────────┘  └──────────┘                               │
 └──────────────────────────────────────────────────────────────┘
```

### Hot-Path Anatomy

```
  CPU (producer)                          GPU persistent worker
  ═══════════════                          ══════════════════════
  ring_acquire()                           ┌─ poll producer tail
       │                                   │   (acquire-load)
       ▼                                   ▼
  write descriptor ──store──▶  ring  ──load──▶  read descriptor
       │                  slot  │                 │
       ▼                       │                 ▼
  ring_commit()                │            read KV arena
  (release-store tail)         │            (hugepage, no TLB miss)
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
  (acquire-load tail)                         (release-store tail)
```

## Components

| Component | File | Description |
|---|---|---|
| Work Descriptor | `include/descriptor.h` | 28-byte packed decode-step command |
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
| Hot-Path Guard | `include/hotpath_guard.h`, `src/hotpath_guard.c` | Wrapper-based tracking for explicit malloc/cudaMalloc/page-fault hooks |
| Copy-on-Write Prefix KV | `include/kv_prefix.h`, `src/kv_prefix.c`, `cu/kv_prefix.cu` | Shared prefix KV with per-rollout delta branches |
| Reward Pipeline | `include/reward.h`, `src/reward.c`, `cu/reward.cu` | Mock reward/verifier ring with GPU scoring kernel |
| Pipeline Benchmark | `bench/bench_pipeline.cu` | Full RL pipeline benchmark with hot-path guard verification |
| Trace Pipeline Benchmark | `bench/bench_trace_pipeline.cu` | Pipeline benchmark with nanosecond tracing and latency percentiles |
| Tracing | `include/trace.h`, `src/trace.c` | Ring-buffer trace entries with pair-latency report (p50/p90/p99) |
| Scheduling Policies | `include/pipeline.h`, `src/pipeline.c` | FIFO / shortest-remaining / prefix-sharing scheduling |
| Pipeline Backpressure | `include/pipeline.h`, `src/pipeline.c` | Credit-based flow control per pipeline stage |
| COW Prefix KV Benchmark | `bench/bench_cow_prefix.cu` | Memory savings comparison: COW vs full-duplicate KV |
| Control-Plane Tax Benchmark | `bench/bench_control_tax.cu` | Syscall vs polling vs persistent worker comparison |
| MMIO Helpers | `include/mmio.h`, `src/mmio.c` | Host-side write barrier and MMIO-style doorbell helpers |
| Hardware Descriptor | `include/hw_desc.h` | 64-byte fixed-format hardware-facing inference descriptor |
| Hardware Ring | `include/hw_ring.h`, `src/hw_ring.c` | Cacheline-owned SPSC descriptor ring for command/done queues |
| Hardware Submit API | `include/infer_submit.h`, `src/infer_submit.c` | Host inference submit path that pushes descriptors and rings a doorbell |
| Hardware Worker Simulator | `include/hw_worker_sim.h`, `src/hw_worker_sim.c` | CPU-only device model that consumes command descriptors and posts completions |
| Hardware Fastpath Test | `test/test_hw_ring.c` | CPU-only validation for the descriptor/doorbell/ring path |
| Hardware Fastpath Benchmark | `bench/bench_hw_fastpath.c` | Doorbell batching and worker-sim throughput benchmark |
| GPU Request Ring | `include/request.h` | SPSC request ring (CPU→GPU) + done ring (GPU→CPU) for GPU-resident scheduler |
| GPU Rollout Scheduler | `cu/gpu_scheduler.cu` | Persistent GPU kernel that manages rollout lifecycle without CPU per-token involvement |
| GPU Scheduler Benchmark | `bench/bench_gpu_scheduler.cu` | Benchmark for GPU-resident rollout scheduler |
| GPU Scheduler Docs | `docs/gpu_scheduler.md` | Design doc for GPU-resident rollout progression |

## Implementation Status: Real vs Stub

| Component | Status | What it actually does |
|-----------|--------|----------------------|
| Command/Completion rings | **Real** | Lock-free SPSC with acquire/release atomics, cacheline-padded indices |
| KV arena (hugepage) | **Real** | `mmap MAP_HUGETLB`, bitmap O(1) alloc, pre-faulted |
| `cp.async` prefetch pipeline | **Real** | Triple-buffered async copy device code |
| GPU sampling (xoshiro256**) | **Real** | Device-side PRNG, top-k/p, temperature scaling |
| Persistent decode worker | **Real** | Per-SM warp, ring poll loop, `__nanosleep` yield |
| NUMA binding | **Real** | `mbind(MPOL_BIND)` with `numa_available()` guard |
| Rollout state machine | **Real** | CAS transitions, slab allocator, transition validation |
| Pipeline rings | **Real** | 6 lock-free ID rings with acquire/release |
| Hot-path guards | **Partial** | Counts explicit wrapper calls; useful for regressions, not a whole-process proof |
| Tracing | **Real** | 1M-entry ring buffer, pair-latency matching, p50/p90/p99 |
| Request/Done rings (v0.3) | **Real** | Host+device atomics, GPU resident slot management |
| Fixed128 decode path | **Partial** | Real QK / softmax / V math for one fixed-shape path; the decode kernel now stages KV with all-lane `cp.async` participation and uses tiled online softmax accumulation to fuse score normalization with V accumulation, while the runtime routes descriptors through a tiny weighted model-state block plus query projection |
| Pipeline windows | **Real** | Snapshot helpers expose queue occupancy, stage credit headroom, and suggested batch windows for decode/reward/trajectory stages; pipeline benches now use them to gate batched decode admission |
| Descriptor windows | **Real** | Host-side decode batch helpers prepare and submit grouped descriptor windows with one ring commit instead of one commit per step |
| Device-visible batch contract | **Partial** | Grouped descriptor windows now stamp explicit batch size and batch index onto each descriptor, and the worker uses that metadata to shape local prefetch state |
| `cp.async` prefetch path | **Partial** | The prefetch layer now uses lane-striped 16-byte `cp.async` helpers and explicit commit/wait flow, but broader multistage overlap is still limited |
| KV layout descriptor | **Scaffold** | Concrete fixed128 KV block math, alignment invariants, and offset helpers |
| Attention decoder | **Partial** | Fixed128 real path exists; broader runtime/model coverage is still incomplete |
| Hardware-facing inference fastpath | **Real** | 64-byte descriptors, cacheline-owned command/done rings, MMIO-style doorbell writes, and a CPU-only worker simulator for device-model validation |
| Reward model | **Stub** | `reward_score_mock()` returns `(n & 0xFF) / 255.0f` — no real scoring |

Benchmarks measure **ring throughput, control-plane latency, and pipeline overhead**,
not FLOPs or model quality. The attention stub means token/s numbers reflect the
control-path speed, not actual decode performance.

## What This Is Not

This is not a replacement for vLLM, TensorRT-LLM, SGLang, or JAX.

This is a **reference fast-path** showing how an RL inference runtime
could structure fixed KV ownership, CPU→GPU command rings, persistent
decode workers, hugepage-backed memory, cacheline-aware queues, and
async reward handoff.

The goal is to study the control-plane mechanics, not to outperform
production inference stacks today.

## Hardware-Facing Inference Fastpath

The repository now includes a second host-side fastpath that is shaped
like a real device queue rather than a CUDA-specific control surface.

- each `hw_desc_t` is exactly 64 bytes: one cache line, fixed format,
  no heap pointers
- `hw_ring_t` separates consumer-owned `head` from producer-owned `tail`
  on different cache lines and uses acquire/release atomics
- `infer_submit_decode()` pushes a descriptor, release-stores the tail,
  and rings an MMIO-style doorbell with `mmio_write32()`
- `hw_worker_sim_run()` provides a CPU-only device model that consumes
  command descriptors and posts done/reward-needed completions

This is a hardware-facing model, not direct access to NVIDIA internal
GPU doorbells. The point is to make the queueing discipline, descriptor
shape, and doorbell tradeoffs explicit and measurable.

## RTL Descriptor Engine

The repo now also includes an RTL model of the hardware queue/control-plane
protocol. It mirrors the C hardware-facing fast path:

```text
C infer_submit_decode()
    |
    v
hw_desc_t / command ring
    |
    v
RTL desc_ring
    |
    v
rollout_worker_fsm
    |
    v
completion_ring
    |
    v
host observes completion
```

It models:

- fixed descriptors
- descriptor ring
- MMIO doorbell register
- rollout worker FSM
- completion ring
- ready/valid backpressure

It does not model:

- real transformer kernels
- tensor cores
- GB300 internals
- NVIDIA hardware doorbells

| Layer | Status |
|---|---|
| C/CUDA runtime fast path | implemented / experimental |
| Hardware descriptor format | implemented |
| RTL descriptor engine | implemented as control-plane model |
| RTL worker FSM | fake decode / control-plane simulation |
| RTL transformer compute | not implemented |
| Verilator C co-sim | implemented |

## C + Verilator Co-Simulation

The repo now includes a Verilator bridge that lets the host-side C++
simulation submit descriptors into the RTL descriptor engine and observe
completions.

```text
C++ RtlRuntimeBridge
  -> Verilated rl_runtime_top
  -> RTL desc_ring
  -> RTL rollout_worker_fsm
  -> RTL completion_ring
  -> C++ completion polling
```

Commands:

```bash
make verilate
make sim-rtl
make test-rtl-bridge
```

Target passing output:

```text
RTL co-sim basic decode: PASS
RTL co-sim reward boundary: PASS
RTL co-sim completion backpressure: PASS
RTL bridge tests: PASS
```

| Layer | Status |
|---|---|
| C/CUDA runtime | implemented / experimental |
| Hardware descriptor path | implemented |
| RTL descriptor engine | implemented |
| C + Verilator bridge | implemented |
| Real transformer compute in RTL | not implemented |
| GB300 hardware validation | not implemented |

This co-simulation validates the descriptor/control-plane contract. It
does not validate real GPU execution, GB300 hardware behavior, or
transformer math. The current bridge validates a shared logical
descriptor contract, not yet a byte-identical 64-byte wire format; see
`docs/descriptor_contract.md`.

## Status Honesty

| Layer | Status |
|---|---|
| C descriptor ring | implemented |
| C hardware fast path | implemented |
| COW prefix KV | implemented / benchmarked |
| RL pipeline | implemented / simulated |
| RTL descriptor ring | implemented |
| RTL worker FSM | fake decode / control-plane model |
| RTL transformer compute | not implemented |
| GB300 hardware validation | not implemented |
| C + Verilator bridge | implemented |

## Documentation

| File | What it covers |
|---|---|
| `docs/architecture.md` | Full architecture map, hot path vs init path, component interactions |
| `docs/hotpath.md` | Every operation classified as init vs hot path |
| `docs/metrics.md` | Target metrics and benchmark commands (all benchmarks) |
| `docs/rtl.md` | RTL control-plane scope, module map, ring logic, and roadmap |
| `docs/verilator_bridge.md` | Verilator bridge API, signal mapping, tests, and limitations |
| `docs/descriptor_contract.md` | Current C, RTL, and bridge descriptor/completion contract, plus the 64-byte unification roadmap |
| `docs/tracing.md` | Trace event types, latency pairs, example output |
| `docs/gpu_scheduler.md` | GPU-resident rollout scheduler design and comparison |
| `docs/decode_microkernel.md` | Status and intent of the fixed-shape decode microkernel scaffold |
| `docs/part3-metal-blog.md` | Deep dive on PTX, `cp.async`, memory ordering, host flush semantics, and the v0.4 roadmap |
| `docs/v0.2.2-roadmap.md` | File-by-file plan for the first real hardware-close decode path |
| `rtl/README.md` | How to run and reason about the RTL descriptor engine |
| `docs/release-notes-v0.3-rtl.md` | RTL-focused release-note draft |

## Build

Requires Linux, CUDA 12.x+, and `libnuma-dev` for the full runtime.

```bash
make               # build library + test bench + all benchmarks
make smoke         # CPU-only smoke tests for ring/overflow correctness
make test          # smoke tests + CUDA-backed unit tests where supported
make test-all      # software tests + RTL tests if iverilog is installed
make test-hw-ring  # CPU-only hardware fastpath tests
make rtl-test      # RTL descriptor-engine testbench
make verilate      # generate Verilator model for rl_runtime_top
make sim-rtl       # basic C++/RTL co-simulation run
make test-rtl-bridge # multi-case C++/RTL co-simulation tests
make bench         # benchmark: 1M tokens through ring+worker
make bench-pipeline # benchmark: full RL pipeline with rollouts, state machine, reward, hot-path guards
make bench-trace   # benchmark with nanosecond tracing + latency percentiles
make bench-cow          # COW prefix KV memory savings benchmark
make bench-tax          # control-plane tax comparison (syscall vs polling vs persistent worker)
make bench-hw-fastpath  # hardware-facing descriptor + doorbell batching benchmark
make bench-gpu-scheduler # GPU-resident rollout scheduler (zero CPU per-token work)
make bench-decode       # fixed128 decode microkernel scaffold benchmark
make bench-kv-layout    # KV block-layout scaffold and offset math check
make bench-prefetch     # isolated global->shared staging microbenchmark
make bench-all          # run all benchmarks
make ci-build           # CPU-only build path used by GitHub Actions
make ci-run             # CPU-only verification path used by GitHub Actions
make cuda-compile-check # compile CUDA translation units with nvcc, no GPU execution
make cuda-ptx-check     # emit PTX for core CUDA translation units, no GPU execution
make rtl-clean          # remove RTL simulator artifact
make clean-verilator    # remove generated Verilator files
```

`cuda-compile-check` and `cuda-ptx-check` still require `nvcc`, but they
do not require a running NVIDIA GPU. They are useful when you want syntax,
template, host/device, and PTX-generation validation on a machine that has
the CUDA toolkit installed but no accelerator attached.

## What Each Benchmark Proves

| Benchmark | Command | What it proves |
|-----------|---------|----------------|
| `bench` | `make bench` | Ring + GPU worker baseline throughput (1M tokens through ring, no rollout logic) |
| `bench-pipeline` | `make bench-pipeline` | End-to-end RL rollout flow: alloc → decode → reward → trajectory → done with wrapper-guard reporting and optional page-fault snapshots |
| `bench-trace` | `make bench-trace` | Nanosecond latency breakdown — where pipeline time is spent (p50/p90/p99 for 8 latency pairs) |
| `bench-cow` | `make bench-cow` | Memory saved by shared prefix KV vs full-duplicate per rollout |
| `bench-tax` | `make bench-tax` | Control-plane overhead: eventfd syscall vs userspace polling vs persistent worker (this runtime) |
| `bench-hw-fastpath` | `make bench-hw-fastpath` | 64-byte hardware descriptor submission, MMIO-style doorbell batching, and CPU-only worker-sim completion throughput |
| `bench-gpu-scheduler` | `make bench-gpu-scheduler` | GPU-managed rollout lifecycle — zero CPU per-token work, CPU only sees request/done |
| `bench-decode` | `make bench-decode` | Fixed128 real math path for one staged KV block, benchmarked separately from control-plane costs |
| `bench-kv-layout` | `make bench-kv-layout` | KV block layout invariants, byte offsets, and fixed-shape memory math |
| `bench-prefetch` | `make bench-prefetch` | Isolated global→shared KV staging cost, comparing baseline shared-memory copy against `cp.async` staging |

## Benchmark Snapshot

```
Hardware:
  GPU:      H100 (or your GPU here)
  CPU:      x86-64 / Grace ARM
  OS:       Linux
  CUDA:     12.x
  Build:    make bench-all

Note: token throughput reflects the stub attention (mock completion write).
Real attention kernels would add compute latency; the numbers here measure
the control path only.

Results (run `make bench-all` on your hardware and replace placeholders with dated measurements):
  Ring throughput:                > 50 M ops/s
  Full pipeline tokens/s:         measure on your hardware
  Pipeline rollouts/s:            measure on your hardware
  COW prefix KV memory saved:     > 90% for 10K branches
  Prefetch staging BW:            measure with make bench-prefetch
  Control-plane tax:
    syscall per step:             ~X ns (baseline)
    userspace polling:            ~Y ns (fast, CPU-hungry)
    persistent worker (this):     ~Z ns (fastest, no CPU tax)
  Wrapper-tracked mallocs:        0 (wrapper-clean)
  Post-init page faults:          inspect OS snapshot output
  Per-token kernel launches:      0
```

Run on your hardware and open a PR to update this table.

## Labs

The `lab/` directory contains six self-contained C experiments
that teach the close-to-metal concepts used in the runtime:

| Lab | What it teaches | Runs on |
|---|---|---|
| `01_false_sharing` | Cache line contention — MESI protocol, padding | any Linux |
| `02_spsc_ring` | Lock-free ring buffer from scratch — atomics, memory ordering | any Linux |
| `03_hugepage_tlb` | 4K vs 2M page TLB miss comparison — why hugepages matter | Linux w/ hugepages |
| `04_syscall_vs_poll` | eventfd wakeup vs shared-memory polling — syscall cost | any Linux |
| `05_doorbell_mock` | Producer/consumer with doorbell — device queue model | any Linux |
| `06_memory_ordering` | publication ordering, stale-descriptor windows, release/acquire | any Linux |

Each lab is standalone — `cd lab/01_false_sharing && make run`.

```bash
make labs      # build all labs
make lab-run   # run all labs sequentially
make lab-run-safe # run the CPU-safe labs used in CI
```

## Design Rules

1. **Pre-fault everything** — no runtime page faults
2. **No cudaMalloc after init** — static slab allocation
3. **CPU stays out of the data path** — descriptors only
4. **NUMA-local memory** — `mbind(MPOL_BIND)` on every allocation
5. **Reward is GPU-resident** — no PCIe round-trips for scoring
6. **NVLink-C2C for coordination** — coherent rings, no DMA
7. **Rollouts are hardware-visible state machines** — CAS transitions through 6 pipeline stages, no Python/CPU per-token control
8. **Hot-path wrappers catch explicit zero-alloc regressions** — useful guardrail, not a global proof
9. **Copy-on-write prefix KV** — shared prompt KV across rollouts, only per-rollout deltas allocated
10. **Credit-based backpressure** — every pipeline stage has a max occupancy; `pipeline_try_push` blocks well before ring-full
11. **Multiple scheduling policies** — FIFO, shortest-remaining-first, and prefix-sharing policies for the decode queue
12. **GPU-resident rollout progression** — v0.3 moves the state machine and pipeline to the GPU; CPU only submits requests and drains completions

## License

MIT
