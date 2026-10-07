// Payoff moments and finalization shared by Volterra pricing engines.
#pragma once

#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::volterra {

using PartialMoments = reductions::MomentSums;

static __global__ void finalize_price_kernel(
    const PartialMoments* __restrict__ partial_moments,
    std::size_t partial_count,
    std::size_t path_count,
    std::size_t result_index,
    float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    double sum = 0.0;
    double sumsq = 0.0;
    for (std::size_t partial = threadIdx.x;
         partial < partial_count;
         partial += blockDim.x) {
        sum += partial_moments[partial].sum;
        sumsq += partial_moments[partial].sumsq;
    }
    const reductions::MomentSums total =
        reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        double price = 0.0;
        double standard_error = 0.0;
        reductions::compute_statistics(
            total, path_count, price, standard_error
        );
        prices[result_index] = static_cast<float>(price);
        standard_errors[result_index] = static_cast<float>(standard_error);
    }
}

}  // namespace ai_factory::workbench::volterra
