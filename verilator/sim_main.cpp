#include "rtl_bridge.h"

#include <cstdio>

namespace {

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

int run_basic_decode()
{
    RtlRuntimeBridge bridge;
    RtlCompletion comp{};
    bridge.reset();
    if (!bridge.submit_decode(7, 0, 10, 3))
        return 1;
    if (!wait_for_completion(bridge, &comp, 256))
        return 1;
    if (comp.rollout_id != 7 || comp.status != RTL_COMP_STATUS_DONE ||
        comp.final_seq_len != 10 || comp.reward_id != 3)
        return 1;
    std::puts("RTL co-sim basic decode: PASS");
    return 0;
}

int run_reward_boundary()
{
    RtlRuntimeBridge bridge;
    RtlCompletion comp{};
    bridge.reset();
    if (!bridge.submit_decode(8, 0, 64, 4))
        return 1;
    if (!wait_for_completion(bridge, &comp, 256))
        return 1;
    if (comp.rollout_id != 8 || comp.status != RTL_COMP_STATUS_REWARD_NEEDED ||
        comp.final_seq_len != 32)
        return 1;
    std::puts("RTL co-sim reward boundary: PASS");
    return 0;
}

int run_backpressure()
{
    RtlRuntimeBridge bridge;
    RtlCompletion comp{};
    bridge.reset();
    if (!bridge.submit_decode(9, 8, 12, 5))
        return 1;

    bridge.set_completion_ready(false);
    for (int i = 0; i < 16; i++)
        bridge.tick();

    bridge.set_completion_ready(true);
    if (!wait_for_completion(bridge, &comp, 256))
        return 1;
    if (comp.rollout_id != 9 || comp.status != RTL_COMP_STATUS_DONE ||
        comp.final_seq_len != 12)
        return 1;
    std::puts("RTL co-sim completion backpressure: PASS");
    return 0;
}

} // namespace

int main()
{
    if (run_basic_decode() != 0)
        return 1;
    if (run_reward_boundary() != 0)
        return 1;
    if (run_backpressure() != 0)
        return 1;

    std::puts("RTL bridge tests: PASS");
    return 0;
}
