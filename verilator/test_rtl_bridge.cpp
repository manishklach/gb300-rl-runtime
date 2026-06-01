#include "rtl_bridge.h"

#include <cstdio>

namespace {

void dump_completion(const char *label, const RtlCompletion &comp)
{
    std::fprintf(stderr,
                 "%s rollout=%u status=0x%02x final_seq_len=%u reward_id=%u\n",
                 label,
                 static_cast<unsigned>(comp.rollout_id),
                 static_cast<unsigned>(comp.status),
                 static_cast<unsigned>(comp.final_seq_len),
                 static_cast<unsigned>(comp.reward_id));
}

bool wait_for_completion(RtlRuntimeBridge &bridge,
                         RtlCompletion *comp,
                         uint64_t timeout_cycles)
{
    bridge.set_completion_ready(true);
    for (uint64_t i = 0; i < timeout_cycles; i++) {
        if (bridge.poll_completion(comp)) {
            bridge.set_completion_ready(false);
            return true;
        }
        bridge.tick();
    }
    bridge.set_completion_ready(false);
    return false;
}

int test_basic_decode()
{
    RtlRuntimeBridge bridge;
    RtlCompletion comp{};
    bridge.reset();
    if (!bridge.submit_decode(7, 0, 10, 3)) {
        std::fputs("test_basic_decode: submit_decode failed\n", stderr);
        return 1;
    }
    if (!wait_for_completion(bridge, &comp, 256)) {
        std::fputs("test_basic_decode: timed out waiting for completion\n", stderr);
        return 1;
    }
    if (!(comp.rollout_id == 7 && comp.status == RTL_COMP_STATUS_DONE &&
          comp.final_seq_len == 10 && comp.reward_id == 3))
        dump_completion("test_basic_decode: unexpected completion", comp);
    return (comp.rollout_id == 7 && comp.status == RTL_COMP_STATUS_DONE &&
            comp.final_seq_len == 10 && comp.reward_id == 3) ? 0 : 1;
}

int test_reward_boundary()
{
    RtlRuntimeBridge bridge;
    RtlCompletion comp{};
    bridge.reset();
    if (!bridge.submit_decode(8, 0, 64, 4)) {
        std::fputs("test_reward_boundary: submit_decode failed\n", stderr);
        return 1;
    }
    if (!wait_for_completion(bridge, &comp, 256)) {
        std::fputs("test_reward_boundary: timed out waiting for completion\n", stderr);
        return 1;
    }
    if (!(comp.rollout_id == 8 &&
          comp.status == RTL_COMP_STATUS_REWARD_NEEDED &&
          comp.final_seq_len == 32))
        dump_completion("test_reward_boundary: unexpected completion", comp);
    return (comp.rollout_id == 8 &&
            comp.status == RTL_COMP_STATUS_REWARD_NEEDED &&
            comp.final_seq_len == 32) ? 0 : 1;
}

int test_completion_backpressure()
{
    RtlRuntimeBridge bridge;
    RtlCompletion comp{};
    bridge.reset();
    if (!bridge.submit_decode(9, 8, 12, 5)) {
        std::fputs("test_completion_backpressure: submit_decode failed\n", stderr);
        return 1;
    }
    bridge.set_completion_ready(false);
    for (int i = 0; i < 16; i++)
        bridge.tick();
    bridge.set_completion_ready(true);
    if (!wait_for_completion(bridge, &comp, 256)) {
        std::fputs("test_completion_backpressure: timed out waiting for completion\n", stderr);
        return 1;
    }
    if (!(comp.rollout_id == 9 && comp.status == RTL_COMP_STATUS_DONE &&
          comp.final_seq_len == 12))
        dump_completion("test_completion_backpressure: unexpected completion", comp);
    return (comp.rollout_id == 9 && comp.status == RTL_COMP_STATUS_DONE &&
            comp.final_seq_len == 12) ? 0 : 1;
}

int test_multiple_descriptors()
{
    RtlRuntimeBridge bridge;
    RtlCompletion comp{};
    bridge.reset();
    if (!bridge.submit_decode(10, 0, 6, 1)) {
        std::fputs("test_multiple_descriptors: first submit_decode failed\n", stderr);
        return 1;
    }
    if (!bridge.submit_decode(11, 0, 64, 2)) {
        std::fputs("test_multiple_descriptors: second submit_decode failed\n", stderr);
        return 1;
    }
    if (!wait_for_completion(bridge, &comp, 256)) {
        std::fputs("test_multiple_descriptors: timed out waiting for first completion\n", stderr);
        return 1;
    }
    if (comp.rollout_id != 10 || comp.status != RTL_COMP_STATUS_DONE) {
        dump_completion("test_multiple_descriptors: unexpected first completion", comp);
        return 1;
    }
    if (!wait_for_completion(bridge, &comp, 256)) {
        std::fputs("test_multiple_descriptors: timed out waiting for second completion\n", stderr);
        return 1;
    }
    if (comp.rollout_id != 11 || comp.status != RTL_COMP_STATUS_REWARD_NEEDED) {
        dump_completion("test_multiple_descriptors: unexpected second completion", comp);
        return 1;
    }
    return 0;
}

int test_ready_gate()
{
    RtlRuntimeBridge bridge;
    bridge.reset();
    const bool ok = bridge.submit_decode(12, 0, 4, 1);
    if (!ok)
        std::fputs("test_ready_gate: submit_decode failed\n", stderr);
    return ok ? 0 : 1;
}

int test_stop_descriptor()
{
    RtlRuntimeBridge bridge;
    RtlCompletion comp{};
    bridge.reset();
    if (!bridge.submit_stop(99)) {
        std::fputs("test_stop_descriptor: submit_stop failed\n", stderr);
        return 1;
    }
    if (!wait_for_completion(bridge, &comp, 64)) {
        std::fputs("test_stop_descriptor: timed out waiting for completion\n", stderr);
        return 1;
    }
    if (!(comp.rollout_id == 99 && comp.status == RTL_COMP_STATUS_STOPPED))
        dump_completion("test_stop_descriptor: unexpected completion", comp);
    return (comp.rollout_id == 99 && comp.status == RTL_COMP_STATUS_STOPPED) ? 0 : 1;
}

} // namespace

int main()
{
    if (test_basic_decode() != 0) return 1;
    if (test_reward_boundary() != 0) return 1;
    if (test_completion_backpressure() != 0) return 1;
    if (test_multiple_descriptors() != 0) return 1;
    if (test_ready_gate() != 0) return 1;
    if (test_stop_descriptor() != 0) return 1;

    std::puts("RTL bridge tests: PASS");
    return 0;
}
