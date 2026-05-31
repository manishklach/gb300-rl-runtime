# Part 3: Going Metal — PTX, `cp.async`, Memory Ordering, and the Road to v0.4

This is the third deep-dive write-up for `gb300-rl-runtime`.

Part 1 was about the runtime shape: persistent workers, lock-free rings,
and the hot-path discipline.

Part 2 was about the first real decode path and the shift from pure
control-plane scaffolding toward actual math.

Part 3 is about going lower still.

The question here is no longer just:

> Can we build a close-to-metal RL inference runtime in C/CUDA?

It is:

> How close to the hardware are we willing to go, and which parts of the
> stack are still hiding real cost behind pleasant abstractions?

That means talking directly about:

- GPU memory hierarchy
- `cp.async`
- PTX and SASS
- acquire/release ordering on NVIDIA GPUs
- x86 and Grace host-side flush/store-load machinery
- what a serious `v0.4` should prioritize

This doc is intentionally more hardware-facing than the rest of the repo.
It is not a product overview. It is an implementation-minded map for
people who want to inspect what the runtime is really asking the machine
to do.

## Repo Update Since the Last Review

The repository has moved well beyond the early ring-and-worker prototype.

By the time of this write-up:

- `v0.3.0` is tagged
- the repo contains more than twenty components
- there is now a GPU-resident scheduler path beside the older
  CPU-per-token path
- the decode stack has explicit `model_state`, `query_producer`,
  `attention_decode`, prefetch, and grouped descriptor submission stages
- the docs set includes architecture notes, hot-path notes, scheduler
  notes, metrics docs, a decode microkernel write-up, and a Pages site

Important additions compared to the earlier shape include:

- `cu/gpu_scheduler.cu`
- `include/request.h`
- `cu/prefetch.cu`
- `cu/sample.cu`
- `bench/bench_gpu_scheduler.cu`
- `docs/gpu_scheduler.md`

The high-level architectural shift is real:

- `v0.2` still exposes the CPU-per-token decode/control loop
- `v0.3` begins pushing request/done boundaries to the CPU and rollout
  progression onto the GPU

That is a much stronger runtime direction than “we have a CUDA kernel.”

## 1. GPU Memory Hierarchy

If you are trying to understand why the repo keeps obsessing over rings,
arena layout, staging buffers, and shared-memory prefetch, start with the
memory ladder.

```text
 Registers                    ~1 cycle
   |
 Shared memory / L0-like use  ~tens of cycles
   |
 L1 / texture path            ~tens of cycles
   |
 L2                           ~hundreds of cycles
   |
 HBM / device DRAM            ~hundreds to low-thousands of cycles
   |
 NVLink / coherent host path  much slower than HBM, but still viable
   |
 PCIe / pageable host memory  worst case for a decode hot path
```

The exact numbers vary by architecture, occupancy, contention, and
instruction mix, but the shape does not.

What matters for this repo:

1. The decode hot path wants registers and shared memory to carry as much
   of the working set as possible.
2. KV data is too large to “just live in registers,” so layout and
   staging dominate whether the kernel feels memory-bound or compute-bound.
3. CPU-visible control structures must be coherent, ordered, and
   predictable, because even a correct ring becomes useless if visibility
   lags the schedule.

That is why the repo keeps returning to the same pillars:

- fixed-shape KV layout
- pre-allocated KV arena
- persistent workers
- host-visible but tightly controlled ring traffic
- explicit prefetch/staging layers

### Why the KV Arena Matters

The KV arena is not just an allocator convenience. It is a memory-shape
decision.

The ideal decode path wants:

- stable block addresses
- predictable block alignment
- fixed block size
- no hot-path allocation
- no pageable-memory surprises

That lets the decode path reason in terms of:

- block IDs
- stage slots
- vectorized loads
- shared-memory tiles

instead of “some pointer returned by a dynamic subsystem at runtime.”

### Why the Ring Matters

The ring is a latency amplifier if you get it wrong.

Even if the attention kernel is fast, a bad producer/consumer protocol
will leak time in:

- stale tail/head visibility
- polling on the wrong index
- overflow confusion
- expensive host wakeups
- cache or coherence surprises

That is why the repo’s correctness work on ring ownership and acquire /
release semantics was not “boring housekeeping.” It was foundational.

## 2. `cp.async`

If there is one instruction family that marks the transition from
ordinary CUDA code to more hardware-shaped CUDA code, it is `cp.async`.

Conceptually:

- old path: load from global into registers, then store into shared
- `cp.async` path: ask the hardware to move data from global to shared
  asynchronously, without routing the payload through the normal register
  path

That matters because register pressure and memory latency are both enemies
in a decode kernel.

### The Core PTX Shape

At the PTX level, the instruction looks like this:

```ptx
cp.async.ca.shared.global [smem_addr], [gmem_addr], 16;
```

The important parts:

- `cp.async`
  means asynchronous copy
- `.ca`
  means cache at all levels on the way in
- `.shared.global`
  means the transfer is from global memory into shared memory
- `16`
  means the copy granularity is 16 bytes in this form

That last detail matters a lot.

One of the big traps in early low-level CUDA work is treating
`cp.async` like “a magic memcpy instruction for huge buffers.”
It is not.

You build a real prefetch path by issuing many small, lane-striped async
copies and then explicitly committing and waiting on those copy groups.

### What the Repo Does Now

The repo’s newer prefetch path moved toward the correct hardware shape:

- lane-striped 16-byte helper
- explicit commit helper
- explicit wait helper
- stage-pointer helpers for prefetch slots

The relevant abstraction now looks more like:

```c
cp_async_ca_16(smem_dst + off, gmem_src + off);
cp_async_commit();
cp_async_wait_all();
```

instead of a fake one-shot “copy the whole 16 KB block” inline-assembly
placeholder.

That is a meaningful improvement because it matches how the hardware path
actually has to be built.

### The Software Pipeline Pattern

The essential idea is:

1. issue async copies for the next tile or block
2. keep computing on the current tile
3. commit the outstanding copy group
4. wait only when the next stage truly becomes needed

In pipeline form:

```text
 Stage 0: copy tile N+1 into shared
 Stage 1: compute on tile N already resident in shared
 Stage 2: wait only when tile N+1 is needed
```

That is how HBM latency starts getting hidden under useful arithmetic.

### The `__pipeline_wait_prior(1)` Trick

On higher-level CUDA pipeline APIs, one common pattern is:

```c
__pipeline_wait_prior(1);
```

The point is not “wait for everything.”

The point is:

- keep at least one stage in flight
- wait only enough to ensure the oldest needed stage is available

That pattern is important because over-synchronizing a supposedly async
prefetch path destroys the very overlap you were trying to buy.

### What v0.4 Should Improve

The current repo now has better `cp.async` helpers, but it still has more
to do before the path is truly convincing:

- overlap across multiple descriptor-adjacent steps in one batch
- clearer stage ownership in the worker
- proper measurement of prefetch effectiveness, not just existence
- bank-conflict and shared-memory footprint awareness

## 3. PTX and SASS

High-level CUDA tells you what you asked for.
PTX tells you what the compiler decided that meant.
SASS tells you what the GPU will actually execute.

For runtime engineering, PTX and SASS are not academic curiosities.
They are often the only way to verify that the intended memory ordering,
polling scope, or async behavior really survived compilation.

### Example: Ring Tail Poll

At the C/CUDA level, a hot-path poll might look morally like:

```c
uint32_t tail = atomic_load_explicit(&ring->prod.tail, memory_order_acquire);
if (head >= tail)
    __nanosleep(100);
```

The PTX you want to see is in the family of:

```ptx
ld.acquire.gpu.u32 %r4, [%rd_tail];
setp.ge.u32 %p1, %r_head, %r4;
@%p1 nanosleep.u32 100;
```

The key word there is `acquire`.

Without the right ordering qualifier, the consumer may observe the tail
advance before the corresponding descriptor contents are safely visible.

### Example: Rollout State Transition CAS

At the C level:

```c
__atomic_compare_exchange_n(&r->state, &expected, to, 0,
                            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
```

The PTX you want to see is in the family of:

```ptx
atom.cas.acq_rel.gpu.b32 %r_out, [%rd_state], %r_expected, %r_new;
```

Again, the important idea is not memorizing syntax.
It is verifying:

- this is really a CAS
- it really carries acquire/release semantics
- the scope really matches the visibility domain you need

### How to Inspect Your Own Output

For PTX:

```bash
nvcc -arch=sm_90a -ptx cu/worker.cu -o worker.ptx
```

For cubin / SASS:

```bash
nvcc -arch=sm_90a -cubin cu/worker.cu -o worker.cubin
cuobjdump --dump-sass worker.cubin
```

For kernel profiling:

```bash
ncu --set full ./build/bench_decode_microkernel
```

Those tools answer different questions:

- `-ptx`: what virtual ISA the compiler emitted
- `cuobjdump --dump-sass`: what real machine instructions are present
- `ncu`: whether the kernel behaves the way the ISA inspection suggests

### What to Look For

For this repo specifically, interesting inspection targets include:

- ring acquire/release loads and stores
- worker polling loop shape
- `cp.async` emission
- CAS transition code
- register pressure and spills in `attention_decode_step_fixed128`

That is where “hardware-shaped” either becomes real or gets exposed as a
story we told ourselves.

## 4. Memory Ordering

Memory ordering is one of the easiest places to write code that is
locally “obvious” and globally wrong.

This repo is especially sensitive to that because it uses:

- host/device-visible rings
- persistent polling
- lock-free control paths
- explicit producer/consumer ownership

### What `__ATOMIC_RELEASE` Really Means

At a conceptual PTX level, a release-store shape is closer to:

```ptx
fence.release.gpu;
st.relaxed.u32 [addr], value;
```

or equivalent compiler-lowered forms that guarantee:

- earlier writes cannot move after the release point
- the consumer observing the release-acquired publication sees the
  payload as well

The point is not that every compiler literally emits that exact two-line
sequence in all contexts.
The point is the ordering contract:

- payload first
- publication second

### Why Scope Matters: `.gpu` vs `.sys`

On NVIDIA hardware, scope matters.

Broadly:

- `.gpu` scope is about visibility within the GPU domain
- `.sys` scope is about a wider system-visible domain

For purely device-resident synchronization, `.gpu` can be exactly what
you want.

For CPU-visible rings, host-coherent memory, or request/done boundaries,
thinking carefully about whether the ordering and visibility domain is
wide enough is mandatory.

### The Common Silent Corruption Bug

The classic bug shape looks like this:

1. producer writes descriptor fields
2. producer publishes tail with insufficient ordering
3. consumer sees new tail
4. consumer reads a partially visible or stale descriptor

The result is not always a clean crash.
It can be much worse:

- occasionally wrong `seq_id`
- wrong `kv_block_offset`
- mysterious completion mismatches
- impossible benchmark outliers

This is exactly why the repo’s command-ring semantics matter so much:

- producer writes tail
- producer reads head for space checks
- consumer writes head
- consumer reads tail for availability checks

That sounds simple because it should be simple.
But the simplicity only holds if the memory ordering is right.

## 5. x86 and Grace Host Side

GPU runtime work often focuses entirely on PTX and shared memory, but the
host side matters too — especially when the CPU is publishing work to a
GPU-visible ring.

### x86: `CLWB`, `CLFLUSHOPT`, and `SFENCE`

If you are writing to a cache-coherent but persistence- or
visibility-sensitive region on x86, the flush/store fencing family
matters:

- `CLWB`
  writes back cache lines without invalidating them
- `CLFLUSHOPT`
  flushes cache lines with weaker ordering and better batching behavior
- `SFENCE`
  orders prior stores and flushes before later stores become visible

A sketch of a host-side publication pattern can look like:

```c
static inline void ring_commit_x86(volatile uint32_t *tail, uint32_t value) {
    asm volatile("sfence" ::: "memory");
    *tail = value;
}
```

And if explicit cache-line writeback were needed for the descriptor
payload itself, that path could involve:

```c
asm volatile("clwb (%0)" :: "r"(ptr) : "memory");
asm volatile("sfence" ::: "memory");
```

The exact recipe depends on whether the platform gives you coherent
host-device visibility already or expects more explicit flushing.

### AArch64 / Grace: `STLR` and `LDAR`

On AArch64, the acquire/release building blocks are explicit and clean:

- `STLR` for release store
- `LDAR` for acquire load

At a sketch level:

```asm
stlr w1, [x0]
ldar w2, [x0]
```

That maps naturally onto ring publication and observation logic.

### Why Grace-Hopper Changes the Story

Grace-Hopper and Grace-Blackwell class coherence is interesting because
the cost model shifts.

With NVLink-C2C coherence, parts of the “flush, fence, pray the device
sees it the way you intended” story become much simpler.

That does not mean memory ordering disappears.
It means:

- visibility plumbing gets less painful
- cache-management burden shrinks
- ring publication becomes conceptually cleaner

That is one of the reasons this style of runtime is especially
interesting on those systems.

## 6. What v0.4 Should Look Like

If `v0.3.x` was about proving out the runtime shape, then `v0.4` should
be about making the most hardware-sensitive sections feel less like a
well-instrumented prototype and more like an intentional machine-level
design.

Here is the prioritized roadmap.

### P1: Make the Decode Path Harder to Ignore

1. Implement a more serious attention kernel.
   The fixed128 path is real, but still narrow and correctness-first.
   `v0.4` should broaden multi-block handling and deepen the scheduling
   strategy inside the warp.

2. Add a benchmark table that compares:
   - per-step descriptor submission
   - grouped descriptor windows
   - device-visible batch contract
   - future cross-descriptor prefetch behavior

3. Fix the architecture story in the docs and benchmarks so it is always
   obvious which path is:
   - control-plane only
   - real math path
   - synthetic pre-decode scaffolding

### P2: Make the Kernel and Prefetch Path More Credible

4. Add warp specialization.
   Instead of one warp doing everything, split responsibility more
   intentionally:
   - producer/prefetch lanes or warp
   - score lanes
   - reduction or output lanes

5. Guard or tighten the NUMA and host-path assumptions more explicitly.
   The repo is already more honest than before, but `v0.4` should keep
   Linux/POSIX and hardware assumptions crisp.

6. Add a real `cp.async` benchmark.
   Not just “the helper exists,” but:
   - how many bytes are in flight
   - how much overlap is achieved
   - whether shared-memory staging is actually hiding HBM latency

### P3: Tooling and Educational Infrastructure

7. Add a Lab 06 on memory ordering and low-level verification.
   That should cover:
   - acquire/release mistakes
   - PTX inspection workflow
   - CPU-visible ring publication patterns
   - expected failure modes when ordering is wrong

8. Add CI checks where practical.
   CUDA verification will remain hardware-dependent, but:
   - smoke
   - compile-time layout checks
   - docs consistency
   - descriptor/ring invariants
   can and should be automated more aggressively.

### The Target Warp-Specialized Kernel Sketch

The shape to aim at is something closer to:

```text
 Warp or lane group A:
   issue async copies for tile N+1

 Warp or lane group B:
   compute QK for tile N

 Warp or lane group C:
   normalize / accumulate V for tile N

 Shared state:
   rolling max
   rolling denominator
   staged V accumulators
```

That is the kind of design where:

- async staging becomes real overlap
- warp roles become explicit
- score and output paths stop looking like “clever reference code”
  and start looking like an actual throughput-oriented kernel

## Closing

The most encouraging thing about the repo now is that it is no longer
just “a good idea with some CUDA files.”

It has:

- real queue correctness work
- real decode math
- real host/device contract evolution
- real prefetch-path refinement
- real docs and release discipline

But going metal always raises the bar.

Once you start talking about:

- PTX
- `cp.async`
- ordering scopes
- SASS inspection
- host-side flush instructions

you lose the right to stay vague.

That is good.

`gb300-rl-runtime` is now at the point where the next gains will come
less from adding new nouns and more from tightening the hardware truth of
the nouns it already has.

That is exactly where an interesting systems repo should be.
