# Hot-Path Analysis

Every operation in the runtime is classified as **init** (happens once at
startup) or **hot path** (happens every decode step).  The goal is zero
non-decode work in the hot path.

## Operation Table

| Operation                            | Init path | Hot path | Notes |
|--------------------------------------|:---------:|:--------:|-------|
| `mmap` hugepage allocation           |     yes   |    no    | One call per arena at startup |
| `munmap` / deallocation              |     yes   |    no    | At shutdown only |
| `mlock` / memory pinning             |     yes   |    no    | Applied once during `arena_init` |
| `mbind` NUMA binding                 |     yes   |    no    | Applied once per allocation |
| `cudaMalloc`                         |     yes   |    no    | Device state allocated at init |
| `cudaHostAlloc`                      |     yes   |    no    | Coherent ring allocation at init |
| `cudaMemcpy` (large H2D/D2H)         |     yes   |    no    | Model weights, sampling state |
| CUDA kernel launch (persistent)      |     yes   |    no    | `decode_worker<<<>>>` once at init |
| `memset` page fault                  |     yes   |    no    | Touch every page during `arena_init` |
| Ring slot acquire (CAS-free)         |     no    |   yes    | `ring_acquire` — load-acquire head/tail |
| Descriptor write                     |     no    |   yes    | Store descriptor to ring slot |
| Ring slot commit (release-store)     |     no    |   yes    | `ring_commit` — store-release head |
| Ring slot consume (acquire-load)     |     no    |   yes    | `ring_consume` — load-acquire head |
| NUFA store (no coherence)            |     no    |   yes    | Worker writes completion |
| `comp_ring_push` (GPU→CPU)           |     no    |   yes    | GPU release-store on completion |
| `comp_ring_poll` (CPU reads)         |     no    |   yes    | CPU acquire-load on completion |
| `__atomic_thread_fence`              |     no    |   yes    | Ordering between descriptor write and doorbell |
| `cp.async` (HBM→SMEM)               |     no    |   yes    | KV block prefetch in worker |
| Token sampling                       |     no    |   yes    | GPU-resident, no CPU round-trip |
| `malloc` / `free`                    |   allowed |    no    | Zero heap operations in hot path |
| `syscall` (any)                      |   allowed |    no    | Zero kernel entries in hot path |
| Scheduler wakeup                     |   allowed |    no    | GPU self-scheduled via persistent workers |
| Page fault                           |   allowed |    no    | All pages pre-faulted at init |
| TLB miss (hugepages)                 |   allowed |  rarely  | 2M/1G pages → ~512 entries cover 1 GB |

## Invariant

> If the GPU can trigger a page fault, call `malloc`, execute a syscall,
> or wait for a CPU scheduler wakeup in the per-token hot path, the
> design is wrong.
