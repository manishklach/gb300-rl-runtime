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
  ┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │  CPU Control │───▶│  SPSC Command   │───▶│  Persistent GPU │
  │  Plane       │    │  Ring (NVLink-  │    │  Workers        │
  │              │◀───│  C2C coherent)  │◀───│                 │
  └──────────────┘    └─────────────────┘    └─────────────────┘
                              │                       │
                              │               ┌───────┴───────┐
                              │               │  Completion   │
                              │               │  Ring         │
                              │               └───────────────┘
                              │                       │
                        ┌─────▼─────┐         ┌───────▼───────┐
                        │  KV Arena │         │  Reward       │
                        │ (hugepage)│         │  (GPU-res.)   │
                        └───────────┘         └───────────────┘
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
| `docs/hotpath.md` | Every operation classified as init vs hot path |
| `docs/metrics.md` | Target metrics and benchmark commands |

## Build

Requires CUDA 12.x+ and `libnuma-dev`.

```bash
make          # build library + test bench
make test     # run unit tests
make bench    # benchmark: 1M tokens through ring+worker
```

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

## License

MIT
