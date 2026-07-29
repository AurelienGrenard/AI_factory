// Block reductions use FP64 moments for stable Monte Carlo statistics.
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

// Reduce one sum and squared sum from every physical thread in the block.
// The caller guarantees a warp-aligned block and zeroes unused contributions.
__device__ __forceinline__ MomentSums reduce_block(
    double sum,
    double sumsq
) {
    extern __shared__ double shared[];

    // Locate the lane, warp, and number of warps with exact bit operations.
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

// Reduce a fixed register array into a distinct shared-memory output region.
template <std::size_t ValueCount>
__device__ __forceinline__ double* reduce_block_values(
    const double (&values)[ValueCount]
) {
    extern __shared__ double shared[];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;
    const unsigned int warp_count = (blockDim.x + 31U) >> 5U;
    double* const totals = shared + ValueCount * warp_count;

    #pragma unroll
    for (std::size_t value_index = 0U;
         value_index < ValueCount;
         ++value_index) {
        double total = values[value_index];
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xFFFFFFFFU, total, offset);
        }
        if (lane == 0U) {
            shared[value_index * warp_count + warp] = total;
        }
    }
    __syncthreads();

    // The first warp reduces one partial value from every physical warp.
    if (warp == 0U) {
        #pragma unroll
        for (std::size_t value_index = 0U;
             value_index < ValueCount;
             ++value_index) {
            double total = lane < warp_count
                ? shared[value_index * warp_count + lane]
                : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xFFFFFFFFU, total, offset);
            }
            if (lane == 0U) totals[value_index] = total;
        }
    }
    __syncthreads();
    return totals;
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
