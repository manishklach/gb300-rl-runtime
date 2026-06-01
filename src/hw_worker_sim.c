#include "hw_worker_sim.h"

#include <stdint.h>

static int
fake_decode_token(hw_desc_t *desc)
{
    desc->seq_len++;
    return desc->seq_len >= desc->max_tokens;
}

static inline void
hw_worker_pause(void)
{
#if defined(__x86_64__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#endif
}

void
hw_worker_sim_run(hw_worker_sim_t *worker)
{
    while (!__atomic_load_n(worker->stop, __ATOMIC_ACQUIRE)) {
        hw_desc_t desc;

        if (hw_ring_pop(worker->cmdq, &desc) != 0) {
            hw_worker_pause();
            continue;
        }

        if (desc.opcode == DESC_OP_STOP)
            break;

        if (desc.opcode != DESC_OP_DECODE)
            continue;

        while (desc.seq_len < desc.max_tokens) {
            const int done = fake_decode_token(&desc);
            worker->decoded_tokens++;

            if (done) {
                desc.flags |= DESC_FLAG_DONE;
                break;
            }

            if ((desc.seq_len & 31u) == 0u) {
                desc.flags |= DESC_FLAG_NEEDS_REWARD;
                break;
            }
        }

        while (hw_ring_push(worker->doneq, &desc) != 0)
            hw_worker_pause();

        worker->completions++;
    }
}
