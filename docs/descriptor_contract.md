# Descriptor Contract

This document makes the current descriptor and completion contract
explicit across the C runtime, the hardware-facing fast path, the RTL
control-plane model, and the Verilator bridge.

## Why This Exists

The bridge is only meaningful if both sides agree on what a "decode
request" and a "completion" mean. Today the repo has:

- a 64-byte C hardware descriptor, `hw_desc_t`
- a smaller packed RTL descriptor, `desc_t`
- a compact RTL completion, `completion_t`
- a bridge that translates between those encodings

So the current contract is shared semantically, but not yet shared as
one byte-identical wire format.

## Current Descriptor Shapes

### C Hardware Descriptor

Defined in `include/hw_desc.h`.

Properties:

- size: 64 bytes
- alignment: 64 bytes
- intent: one cache line, host/device friendly, future hardware-facing
  submission format

| Field | Width | Notes |
|---|---:|---|
| `opcode` | 16 bits | decode/reward/prefill/stop |
| `flags` | 16 bits | reward-needed/done/COW prefix |
| `rollout_id` | 32 bits | rollout identity |
| `kv_arena_id` | 32 bits | KV arena selector |
| `prefix_id` | 32 bits | shared-prefix selector |
| `kv_offset` | 64 bits | KV base or page offset |
| `delta_offset` | 64 bits | delta/update offset |
| `seq_len` | 32 bits | current sequence length |
| `max_tokens` | 32 bits | decode stop target |
| `reward_model_id` | 32 bits | reward model selection |
| `reserved0` | 32 bits | reserved |
| `user_data` | 64 bits | host correlation hook |
| `checksum_or_cookie` | 64 bits | reserved for integrity or tracking |

### RTL Descriptor

Defined in `rtl/desc_pkg.sv` as `desc_t`.

Properties:

- size: 192 bits
- size in bytes: 24 bytes
- intent: compact control-plane request for the RTL queue/FSM model

| Field | Width | Notes |
|---|---:|---|
| `opcode` | 8 bits | `DESC_OP_*` in RTL |
| `flags` | 8 bits | compact control flags |
| `rollout_id` | 16 bits | narrower than C |
| `kv_arena_id` | 16 bits | narrower than C |
| `prefix_id` | 16 bits | narrower than C |
| `kv_offset` | 32 bits | narrower than C |
| `delta_offset` | 32 bits | narrower than C |
| `seq_len` | 16 bits | narrower than C |
| `max_tokens` | 16 bits | narrower than C |
| `reward_model_id` | 16 bits | narrower than C |
| `reserved` | 16 bits | reserved |

## Current Completion Shapes

### RTL Completion

Defined in `rtl/desc_pkg.sv` as `completion_t`.

Properties:

- size: 56 bits
- size in bytes: 7 bytes
- intent: compact completion result for done/reward/stop outcomes

| Field | Width | Notes |
|---|---:|---|
| `rollout_id` | 16 bits | completion identity |
| `status` | 8 bits | done/reward-needed/stopped |
| `final_seq_len` | 16 bits | terminal or boundary length |
| `reward_id` | 16 bits | reward routing payload |

### C Hardware Fastpath Completion

The current C hardware fast path does not yet have a distinct compact
completion struct. The CPU-only worker simulator pushes an updated
`hw_desc_t` into the done ring and the host interprets:

- `DESC_FLAG_DONE`
- `DESC_FLAG_NEEDS_REWARD`
- `seq_len`
- `rollout_id`

That is convenient for software bring-up, but it is not yet the same
shape as the RTL completion contract.

## Opcode Values

Current opcode meanings:

| Meaning | C `include/hw_desc.h` | RTL `rtl/desc_pkg.sv` |
|---|---:|---:|
| NOP | not present | `0` |
| DECODE | `1` | `1` |
| REWARD | `2` | `2` |
| PREFILL | `3` | not present |
| STOP | `255` | `255` |

Notes:

- `DECODE` and `STOP` already line up cleanly.
- `REWARD` lines up numerically.
- `PREFILL` exists only on the C side today.
- `NOP` exists only on the RTL side today.

## Completion Status Values

Current completion-result meanings:

| Meaning | RTL completion status |
|---|---:|
| DONE | `0x01` |
| REWARD_NEEDED | `0x02` |
| STOPPED | `0xff` |

The C hardware fast path currently expresses these states through
descriptor flags and stop opcodes rather than a separate status enum.

## Bridge Mapping

The Verilator bridge maps host intent into RTL fields with narrowing
where required:

- `rollout_id`: C/host values are truncated to 16 bits for RTL
- `kv_arena_id`: truncated to 16 bits
- `prefix_id`: truncated to 16 bits
- `kv_offset`: truncated to 32 bits
- `delta_offset`: truncated to 32 bits
- `seq_len`: truncated to 16 bits
- `max_tokens`: truncated to 16 bits
- `reward_model_id`: truncated to 16 bits

That is acceptable for today's control-plane tests because the test
inputs stay within the narrower RTL ranges. It is also the clearest sign
that the physical descriptor contract is not unified yet.

## Endianness and Packing Assumptions

Current assumptions:

- the C descriptor uses fixed-width integer fields in declaration order
- the C struct is `packed` and `aligned(64)`
- the RTL descriptor uses SystemVerilog packed-struct bit ordering
- the bridge packs fields explicitly rather than relying on host memory
  layout matching RTL bit layout

Implications:

- the repo does not currently claim a host-memory image of `hw_desc_t`
  can be DMA'd directly into the RTL `desc_t` interface
- endianness-sensitive behavior is intentionally isolated in the bridge's
  explicit pack/unpack logic

## What Is Shared Today

The following parts of the contract are already shared semantically:

- decode vs reward vs stop intent
- rollout identity
- KV arena / prefix routing intent
- sequence-length progression
- reward-needed vs done completion meaning
- backpressure and queue-full behavior

## Gaps To Close

To reach a truly shared hardware/software descriptor contract, the repo
still needs:

1. A byte-identical descriptor definition used by both the software
   fast path and the RTL boundary.
2. A byte-identical completion definition used by both the software done
   ring and the RTL completion output.
3. One shared opcode/status namespace with no C-only or RTL-only values.
4. Explicit tests that round-trip descriptor bytes through both the C
   fast path and the RTL bridge without field loss or truncation.

## 64-Byte / 512-Bit Roadmap

The next strong milestone is a single 64-byte descriptor contract:

- size: 64 bytes
- width: 512 bits
- alignment: one cache line
- fixed-width fields
- explicit reserved space for future scheduling, priority, checksum, or
  tracing metadata

Target properties:

- software submit path writes one cache line per descriptor
- hardware/RTL queue ingress consumes one 512-bit descriptor contract
- bridge can switch from semantic translation to byte-accurate transport

That future state would justify the statement:

> The software and RTL layers share one 64-byte hardware descriptor
> contract.

Today, the honest statement is:

> The software and RTL layers share one control-plane idea and one
> logical descriptor contract, but not yet one byte-identical wire
> format.
