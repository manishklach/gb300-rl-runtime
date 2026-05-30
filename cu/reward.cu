#include "reward.h"
#include <cuda_runtime.h>

__global__ void
reward_score_kernel(RewardDesc *descs, int n, float *scores)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    RewardDesc d = descs[i];
    float score = (float)(d.token_count & 0xFF) / 255.0f;
    descs[i].reward = score;
    if (scores)
        scores[i] = score;
}
