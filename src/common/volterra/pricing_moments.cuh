// Payoff moments and finalization shared by Volterra pricing engines.
#pragma once

#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::volterra {

using PartialMoments = reductions::MomentSums;

__device__ __forceinline__ void finalize_price_moments(
    const PartialMoments* __restrict__ partial_moments,
    std::size_t partial_count,std::size_t path_count,
    std::size_t result_index,float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    double sum=0.0,sumsq=0.0;
    for(std::size_t partial=threadIdx.x;partial<partial_count;
        partial+=blockDim.x) {
        sum+=partial_moments[partial].sum;
        sumsq+=partial_moments[partial].sumsq;
    }
    const auto total=reductions::reduce_block(sum,sumsq);
    if(threadIdx.x==0U) {
        double price=0.0,error=0.0;
        reductions::compute_statistics(total,path_count,price,error);
        prices[result_index]=static_cast<float>(price);
        standard_errors[result_index]=static_cast<float>(error);
    }
}

static __global__ void finalize_price_kernel(
    const PartialMoments* __restrict__ partial_moments,
    std::size_t partial_count,std::size_t path_count,
    std::size_t result_index,float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    finalize_price_moments(partial_moments,partial_count,path_count,
        result_index,prices,standard_errors);
}

// One block per price row; partial moments are contiguous within each row.
static __global__ void finalize_price_batch_kernel(
    const PartialMoments* __restrict__ partial_moments,
    std::size_t partial_count,std::size_t path_count,
    std::size_t result_offset,float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    finalize_price_moments(
        partial_moments+blockIdx.x*partial_count,
        partial_count,path_count,result_offset+blockIdx.x,
        prices,standard_errors);
}

}  // namespace ai_factory::workbench::volterra
