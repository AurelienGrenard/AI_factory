// Block-level Monte Carlo reduction helpers shared by pricing kernels.
// Individual payoffs are simulated in FP32, while moments and final statistics
// are accumulated in FP64 to limit summation and cancellation errors.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

#include <cstddef>

namespace ai_factory::workbench::reductions {

// MomentSums carries the first two raw payoff moments for one result row.
struct MomentSums {
    double sum;
    double sumsq;
};

// Reduce one sum and one squared sum across a block.
//
// Contract:
// - every physical thread in the block must call this function;
// - blockDim.x must be a multiple of the 32-thread CUDA warp size;
// - each thread must provide its own contribution in sum and sumsq;
// - if a caller has fewer logical values than physical threads, that caller
//   must pass 0.0 for the threads without a value.
//
// Warp shuffles avoid shared memory for the first stage; only one pair per warp
// is stored in shared memory.
__device__ __forceinline__ MomentSums reduce_block(
    double sum,
    double sumsq
) {
    extern __shared__ double shared[];

    // A CUDA warp contains 32 threads. These bit operations are exact
    // shorthand for:
    //   lane       = threadIdx.x % 32
    //   warp       = threadIdx.x / 32
    //   warp_count = ceil(blockDim.x / 32)
    // They identify the thread inside its warp, the warp inside the block, and
    // the number of warps participating in this block reduction.
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;
    const unsigned int warp_count = (blockDim.x + 31U) >> 5U;
    double* warp_sums = shared;
    double* warp_sumsq = shared + warp_count;

    // First stage: each warp reduces its 32 register-resident contributions.
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
        sumsq += __shfl_down_sync(0xFFFFFFFFU, sumsq, offset);
    }
    if (lane == 0U) {
        warp_sums[warp] = sum;
        warp_sumsq[warp] = sumsq;
    }
    __syncthreads();

    // Second stage: the first warp reduces the one partial pair from each warp.
    if (warp == 0U) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0;
        sumsq = lane < warp_count ? warp_sumsq[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
            sumsq += __shfl_down_sync(0xFFFFFFFFU, sumsq, offset);
        }
        if (lane == 0U) {
            warp_sums[0] = sum;
            warp_sumsq[0] = sumsq;
        }
    }
    __syncthreads();
    return {warp_sums[0], warp_sumsq[0]};
}

// Convert the final FP64 sums for one row into its Monte Carlo statistics.
// Only thread 0 calls this helper after the block reduction has completed.
__device__ __forceinline__ void compute_statistics(
    const MomentSums& total,
    std::size_t sample_count,
    double& price,
    double& standard_error
) {
    const double count = static_cast<double>(sample_count);
    price = total.sum / count;
    const double sample_variance =
        (total.sumsq - count * price * price) / (count - 1.0);
    standard_error = sqrt(fmax(sample_variance, 0.0) / count);
}

}  // namespace ai_factory::workbench::reductions
