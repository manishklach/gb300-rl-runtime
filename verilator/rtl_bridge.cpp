#include "rtl_bridge.h"

#include "Vrl_runtime_top.h"
#include "verilated.h"

#if defined(ENABLE_VCD)
#include "verilated_vcd_c.h"
#endif

#include <cstring>

namespace {

constexpr int kDescWords = 6;
constexpr int kCompWords = 2;

static uint64_t bit_mask(int width)
{
    if (width >= 64)
        return ~0ULL;
    return (1ULL << width) - 1ULL;
}

static uint64_t extract_bits(QData value, int lsb, int width)
{
    uint64_t sliced = ((uint64_t)value) >> lsb;
    if (width < 64)
        sliced &= bit_mask(width);
    return sliced;
}

static uint64_t extract_bits(const VlWide<kCompWords> &words, int lsb, int width)
{
    const int word = lsb / 32;
    const int shift = lsb % 32;
    uint64_t value = ((uint64_t)words[word]) >> shift;
    if (shift + width > 32 && word + 1 < kCompWords)
        value |= ((uint64_t)words[word + 1]) << (32 - shift);
    if (width < 64)
        value &= bit_mask(width);
    return value;
}

} // namespace

RtlRuntimeBridge::RtlRuntimeBridge()
    : context_(new VerilatedContext()),
      top_(new Vrl_runtime_top(context_.get())),
      cycles_(0)
{
    context_->traceEverOn(true);
#if defined(ENABLE_VCD)
    trace_.reset(new VerilatedVcdC());
    top_->trace(trace_.get(), 5);
    trace_->open("build/rtl_bridge.vcd");
#endif
    top_->clk = 0;
    top_->rst_n = 0;
    top_->host_desc_valid = 0;
    top_->host_comp_ready = 0;
    top_->doorbell_pulse = 0;
    for (int i = 0; i < kDescWords; i++)
        top_->host_desc[i] = 0;
    eval();
}

RtlRuntimeBridge::~RtlRuntimeBridge()
{
#if defined(ENABLE_VCD)
    if (trace_)
        trace_->close();
#endif
}

void RtlRuntimeBridge::eval()
{
    top_->eval();
#if defined(ENABLE_VCD)
    if (trace_)
        trace_->dump(context_->time());
#endif
}

void RtlRuntimeBridge::tick()
{
    top_->clk = 0;
    eval();
    context_->timeInc(1);

    top_->clk = 1;
    eval();
    context_->timeInc(1);

    cycles_++;
}

void RtlRuntimeBridge::reset(unsigned cycles)
{
    top_->rst_n = 0;
    top_->host_desc_valid = 0;
    top_->host_comp_ready = 0;
    top_->doorbell_pulse = 0;
    for (unsigned i = 0; i < cycles; i++)
        tick();
    top_->rst_n = 1;
    tick();
}

void RtlRuntimeBridge::pack_desc(uint8_t opcode,
                                 uint8_t flags,
                                 uint16_t rollout_id,
                                 uint16_t kv_arena_id,
                                 uint16_t prefix_id,
                                 uint32_t kv_offset,
                                 uint32_t delta_offset,
                                 uint16_t seq_len,
                                 uint16_t max_tokens,
                                 uint16_t reward_model_id,
                                 uint16_t reserved)
{
    for (int i = 0; i < kDescWords; i++)
        top_->host_desc[i] = 0;

    top_->host_desc[0] =
        (uint32_t)opcode |
        ((uint32_t)flags << 8) |
        ((uint32_t)rollout_id << 16);
    top_->host_desc[1] =
        (uint32_t)kv_arena_id |
        ((uint32_t)prefix_id << 16);
    top_->host_desc[2] = kv_offset;
    top_->host_desc[3] = delta_offset;
    top_->host_desc[4] =
        (uint32_t)seq_len |
        ((uint32_t)max_tokens << 16);
    top_->host_desc[5] =
        (uint32_t)reward_model_id |
        ((uint32_t)reserved << 16);
}

bool RtlRuntimeBridge::host_desc_ready() const
{
    return top_->host_desc_ready != 0;
}

bool RtlRuntimeBridge::host_comp_valid() const
{
    return top_->host_comp_valid != 0;
}

bool RtlRuntimeBridge::submit_decode(uint16_t rollout_id,
                                     uint16_t seq_len,
                                     uint16_t max_tokens,
                                     uint16_t reward_model_id,
                                     uint16_t kv_arena_id,
                                     uint16_t prefix_id,
                                     uint32_t kv_offset,
                                     uint32_t delta_offset)
{
    pack_desc(RTL_DESC_OP_DECODE, 0u, rollout_id, kv_arena_id, prefix_id,
              kv_offset, delta_offset, seq_len, max_tokens, reward_model_id, 0u);

    if (!host_desc_ready()) {
        tick();
        if (!host_desc_ready())
            return false;
    }

    top_->host_desc_valid = 1;
    top_->doorbell_pulse = 1;
    tick();
    top_->host_desc_valid = 0;
    top_->doorbell_pulse = 0;
    return true;
}

bool RtlRuntimeBridge::submit_stop(uint16_t rollout_id)
{
    pack_desc(RTL_DESC_OP_STOP, 0u, rollout_id, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    if (!host_desc_ready()) {
        tick();
        if (!host_desc_ready())
            return false;
    }

    top_->host_desc_valid = 1;
    top_->doorbell_pulse = 1;
    tick();
    top_->host_desc_valid = 0;
    top_->doorbell_pulse = 0;
    return true;
}

void RtlRuntimeBridge::set_completion_ready(bool ready)
{
    top_->host_comp_ready = ready ? 1 : 0;
}

bool RtlRuntimeBridge::poll_completion(RtlCompletion *out)
{
    if (!host_comp_valid() || !out)
        return false;

    out->rollout_id = (uint16_t)extract_bits(top_->host_comp, 0, 16);
    out->status = (uint8_t)extract_bits(top_->host_comp, 16, 8);
    out->final_seq_len = (uint16_t)extract_bits(top_->host_comp, 24, 16);
    out->reward_id = (uint16_t)extract_bits(top_->host_comp, 40, 16);

    if (top_->host_comp_ready)
        tick();
    return true;
}

uint64_t RtlRuntimeBridge::cycles() const
{
    return cycles_;
}
