# GB300 RL Inference Runtime

A close-to-metal C/CUDA reference runtime for reinforcement learning
inference at GB300 NVL72 scale.  No page faults, no `malloc`/`free`,
no kernel launches, no CPU scheduler wakeups in the per-token hot path.

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
