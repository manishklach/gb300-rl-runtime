#pragma once

#include <cstdint>
#include <memory>

struct RtlCompletion {
    uint16_t rollout_id;
    uint8_t status;
    uint16_t final_seq_len;
    uint16_t reward_id;
};

class VerilatedContext;
class VerilatedVcdC;
class Vrl_runtime_top;

class RtlRuntimeBridge {
public:
    RtlRuntimeBridge();
    ~RtlRuntimeBridge();

    void reset(unsigned cycles = 5);
    void tick();

    bool submit_decode(uint16_t rollout_id,
                       uint16_t seq_len,
                       uint16_t max_tokens,
                       uint16_t reward_model_id,
                       uint16_t kv_arena_id = 0,
                       uint16_t prefix_id = 0,
                       uint32_t kv_offset = 0,
                       uint32_t delta_offset = 0);

    bool submit_stop(uint16_t rollout_id = 0);
    bool poll_completion(RtlCompletion *out);

    void set_completion_ready(bool ready);
    uint64_t cycles() const;

private:
    void eval();
    void pack_desc(uint8_t opcode,
                   uint8_t flags,
                   uint16_t rollout_id,
                   uint16_t kv_arena_id,
                   uint16_t prefix_id,
                   uint32_t kv_offset,
                   uint32_t delta_offset,
                   uint16_t seq_len,
                   uint16_t max_tokens,
                   uint16_t reward_model_id,
                   uint16_t reserved);
    bool host_desc_ready() const;
    bool host_comp_valid() const;

    std::unique_ptr<VerilatedContext> context_;
    std::unique_ptr<Vrl_runtime_top> top_;
#if defined(ENABLE_VCD)
    std::unique_ptr<VerilatedVcdC> trace_;
#endif
    uint64_t cycles_;
};

constexpr uint8_t RTL_DESC_OP_DECODE = 1u;
constexpr uint8_t RTL_DESC_OP_STOP = 0xffu;
constexpr uint8_t RTL_COMP_STATUS_DONE = 0x01u;
constexpr uint8_t RTL_COMP_STATUS_REWARD_NEEDED = 0x02u;
constexpr uint8_t RTL_COMP_STATUS_STOPPED = 0xffu;
