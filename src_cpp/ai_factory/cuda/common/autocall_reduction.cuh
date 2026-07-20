#pragma once

#include "ai_factory/cuda/common/autocall.cuh"
#include "ai_factory/cuda/common/reductions.cuh"

#include <cmath>
#include <cstddef>

namespace ai_factory::cuda::autocall_detail {

constexpr int kMetricCount = 8;

__device__ __forceinline__ void metrics_to_values(
    const PathMetrics& metrics,
    double (&values)[kMetricCount]
) {
    values[0] = metrics.discounted_payoff;
    values[1] = metrics.discounted_payoff * metrics.discounted_payoff;
    values[2] = metrics.autocall;
    values[3] = metrics.autocall_time;
    values[4] = metrics.coupon_payment_frequency;
    values[5] = metrics.total_coupon;
    values[6] = metrics.capital_loss;
    values[7] = metrics.loss_redemption;
}

__device__ __forceinline__ void reduce_and_store(
    double (&values)[kMetricCount],
    std::size_t partial_index,
    std::size_t partial_count,
    double* partials
) {
    reductions::reduce_block_values(values);
    if (threadIdx.x == 0U) {
        extern __shared__ double shared[];
#pragma unroll
        for (int metric = 0; metric < kMetricCount; ++metric) {
            partials[static_cast<std::size_t>(metric) * partial_count + partial_index] =
                shared[static_cast<std::size_t>(metric) * blockDim.x];
        }
    }
}

static __global__ void autocall_finalize_kernel(
    const double* partials,
    std::size_t row_count,
    std::size_t path_blocks_per_row,
    std::size_t num_paths,
    AutocallOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }
    const auto partial_count = row_count * path_blocks_per_row;
    double values[kMetricCount]{};
    for (std::size_t block = threadIdx.x;
         block < path_blocks_per_row;
         block += blockDim.x) {
        const auto partial_index = row_index * path_blocks_per_row + block;
#pragma unroll
        for (int metric = 0; metric < kMetricCount; ++metric) {
            values[metric] +=
                partials[static_cast<std::size_t>(metric) * partial_count + partial_index];
        }
    }
    reductions::reduce_block_values(values);
    if (threadIdx.x == 0U) {
        extern __shared__ double shared[];
        const double count = static_cast<double>(num_paths);
        const double price = shared[0] / count;
        const double variance =
            (shared[blockDim.x] - count * price * price)
            / static_cast<double>(num_paths - 1U);
        const double autocall_count = shared[2U * blockDim.x];
        const double loss_count = shared[6U * blockDim.x];
        const double autocall_probability = autocall_count / count;
        outputs[row_index] = {
            price,
            sqrt(fmax(variance, 0.0)) / sqrt(count),
            autocall_probability,
            autocall_count > 0.0
                ? shared[3U * blockDim.x] / autocall_count
                : 0.0,
            1.0 - autocall_probability,
            shared[4U * blockDim.x] / count,
            shared[5U * blockDim.x] / count,
            loss_count / count,
            loss_count > 0.0
                ? shared[7U * blockDim.x] / loss_count
                : 0.0,
        };
    }
}

}  // namespace ai_factory::cuda::autocall_detail
