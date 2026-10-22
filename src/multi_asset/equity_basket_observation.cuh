// Basket statistics reduced across the threads owning one joint path.
#pragma once

#include "correlation/correlated_brownian.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::equity {

// The basket is built from S_i(t) / S_i(0), so its initial value is one.
enum class BasketSpotAggregation {
    Minimum,
    Maximum,
    ArithmeticMean
};

template<BasketSpotAggregation Aggregation, typename Pack, unsigned TeamSize>
__device__ __forceinline__ float team_basket_spot_ratio(
    const typename Pack::States& states,
    const typename Pack::Prepared& prepared,
    unsigned lane
) {
    float value;
    if constexpr (Aggregation == BasketSpotAggregation::Minimum)
        value = Pack::template minimum_ratio<0U, TeamSize>(
            states, prepared, lane);
    else if constexpr (Aggregation == BasketSpotAggregation::Maximum)
        value = Pack::template maximum_ratio<0U, TeamSize>(
            states, prepared, lane);
    else
        value = Pack::template sum_ratio<0U, TeamSize>(
            states, prepared, lane);

    if constexpr (TeamSize > 1U) {
        const unsigned mask = simulation::subgroup_mask<TeamSize>();
        for (unsigned offset = TeamSize / 2U; offset > 0U; offset >>= 1U) {
            const float other = __shfl_down_sync(
                mask, value, offset, TeamSize);
            if constexpr (Aggregation == BasketSpotAggregation::Minimum)
                value = fminf(value, other);
            else if constexpr (Aggregation == BasketSpotAggregation::Maximum)
                value = fmaxf(value, other);
            else
                value += other;
        }
    }
    if constexpr (Aggregation == BasketSpotAggregation::ArithmeticMean)
        value /= static_cast<float>(Pack::kAssetCount);
    return value;
}

template<BasketSpotAggregation Aggregation, typename Pack,
         std::size_t RoughCount, unsigned TeamSize>
__device__ __forceinline__ float team_joint_basket_spot_ratio(
    const typename Pack::States& states,
    const typename Pack::Prepared& prepared,
    const float (&rough_logs)[RoughCount],
    const float (&rough_initial)[RoughCount],
    unsigned lane
) {
    float value;
    if constexpr (Aggregation == BasketSpotAggregation::Minimum)
        value = Pack::template minimum_ratio<0U, TeamSize>(
            states, prepared, lane);
    else if constexpr (Aggregation == BasketSpotAggregation::Maximum)
        value = Pack::template maximum_ratio<0U, TeamSize>(
            states, prepared, lane);
    else
        value = Pack::template sum_ratio<0U, TeamSize>(
            states, prepared, lane);

    for (std::size_t i = 0U; i < RoughCount; ++i) {
        if (i % TeamSize != lane) continue;
        const float rough_ratio = expf(rough_logs[i] - rough_initial[i]);
        if constexpr (Aggregation == BasketSpotAggregation::Minimum)
            value = fminf(value, rough_ratio);
        else if constexpr (Aggregation == BasketSpotAggregation::Maximum)
            value = fmaxf(value, rough_ratio);
        else
            value += rough_ratio;
    }

    if constexpr (TeamSize > 1U) {
        const unsigned mask = simulation::subgroup_mask<TeamSize>();
        for (unsigned offset = TeamSize / 2U; offset > 0U; offset >>= 1U) {
            const float other = __shfl_down_sync(
                mask, value, offset, TeamSize);
            if constexpr (Aggregation == BasketSpotAggregation::Minimum)
                value = fminf(value, other);
            else if constexpr (Aggregation == BasketSpotAggregation::Maximum)
                value = fmaxf(value, other);
            else
                value += other;
        }
    }
    if constexpr (Aggregation == BasketSpotAggregation::ArithmeticMean)
        value /= static_cast<float>(Pack::kAssetCount + RoughCount);
    return value;
}

}  // namespace ai_factory::workbench::equity
