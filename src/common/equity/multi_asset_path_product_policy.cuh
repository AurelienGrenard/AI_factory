// Equity observables reduced across the threads owning one joint path.
#pragma once

#include "common/simulation/correlated_brownian.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::equity {

template<typename Pack, unsigned int TeamSize>
__device__ __forceinline__ float team_minimum_spot_ratio(
    const typename Pack::States& states,
    const typename Pack::Prepared& prepared,
    unsigned int lane
) {
    float minimum = Pack::template minimum_ratio<0U, TeamSize>(
        states, prepared, lane
    );
    if constexpr (TeamSize > 1U) {
        const unsigned int mask = simulation::subgroup_mask<TeamSize>();
        for (unsigned int offset = TeamSize / 2U; offset > 0U;
             offset >>= 1U) {
            minimum = fminf(
                minimum,
                __shfl_down_sync(mask, minimum, offset, TeamSize)
            );
        }
    }
    return minimum;
}

template<typename Pack, unsigned int TeamSize>
__device__ __forceinline__ float team_sum_spot_ratio(
    const typename Pack::States& states,
    const typename Pack::Prepared& prepared,
    unsigned int lane
) {
    float sum = Pack::template sum_ratio<0U, TeamSize>(
        states, prepared, lane
    );
    if constexpr (TeamSize > 1U) {
        const unsigned int mask = simulation::subgroup_mask<TeamSize>();
        for (unsigned int offset = TeamSize / 2U; offset > 0U;
             offset >>= 1U) {
            sum += __shfl_down_sync(mask, sum, offset, TeamSize);
        }
    }
    return sum;
}


template<typename Pack, std::size_t RoughCount, unsigned TeamSize>
__device__ __forceinline__ float team_joint_minimum_spot_ratio(
    const typename Pack::States& states,
    const typename Pack::Prepared& prepared,
    const float (&rough_logs)[RoughCount],
    const float (&rough_initial)[RoughCount],
    unsigned lane
) {
    float value=Pack::template minimum_ratio<0U,TeamSize>(
        states,prepared,lane);
    for(std::size_t i=0U;i<RoughCount;++i)
        if(i%TeamSize==lane)
            value=fminf(value,expf(rough_logs[i]-rough_initial[i]));
    if constexpr (TeamSize>1U) {
        const auto mask=simulation::subgroup_mask<TeamSize>();
        for(unsigned offset=TeamSize/2U;offset>0U;offset>>=1U)
            value=fminf(value,__shfl_down_sync(mask,value,offset,TeamSize));
    }
    return value;
}

template<typename Pack, std::size_t RoughCount, unsigned TeamSize>
__device__ __forceinline__ float team_joint_sum_spot_ratio(
    const typename Pack::States& states,
    const typename Pack::Prepared& prepared,
    const float (&rough_logs)[RoughCount],
    const float (&rough_initial)[RoughCount],
    unsigned lane
) {
    float value=Pack::template sum_ratio<0U,TeamSize>(
        states,prepared,lane);
    for(std::size_t i=0U;i<RoughCount;++i)
        if(i%TeamSize==lane)
            value+=expf(rough_logs[i]-rough_initial[i]);
    if constexpr (TeamSize>1U) {
        const auto mask=simulation::subgroup_mask<TeamSize>();
        for(unsigned offset=TeamSize/2U;offset>0U;offset>>=1U)
            value+=__shfl_down_sync(mask,value,offset,TeamSize);
    }
    return value;
}

}  // namespace ai_factory::workbench::equity
