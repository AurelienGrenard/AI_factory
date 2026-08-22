// Reusable path-local accumulators evaluated at contractual observations.
#pragma once

#include "common/equity/concepts.cuh"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::equity {

template<EquityDynamicsPolicy Dynamics>
struct ArithmeticMeanObservationHandler {
    double spot_sum = 0.0;
    std::uint32_t observation_count = 0U;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return on_observation(0U, state);
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        spot_sum += static_cast<double>(Dynamics::spot(state));
        ++observation_count;
        return true;
    }

    __device__ __forceinline__ float arithmetic_mean() const {
        return observation_count == 0U
            ? 0.0f
            : static_cast<float>(
                spot_sum / static_cast<double>(observation_count)
            );
    }
};

template<SupportsLogSpot Dynamics>
struct GeometricMeanObservationHandler {
    double log_spot_sum = 0.0;
    std::uint32_t observation_count = 0U;
    bool hit_non_positive_spot = false;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return on_observation(0U, state);
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        const float spot = Dynamics::spot(state);
        hit_non_positive_spot =
            hit_non_positive_spot || !(spot > 0.0f);
        if (!hit_non_positive_spot) {
            log_spot_sum += static_cast<double>(
                Dynamics::log_spot(state)
            );
        }
        ++observation_count;
        return true;
    }

    __device__ __forceinline__ float geometric_mean() const {
        if (observation_count == 0U || hit_non_positive_spot) return 0.0f;
        return expf(static_cast<float>(
            log_spot_sum / static_cast<double>(observation_count)
        ));
    }
};

template<EquityDynamicsPolicy Dynamics>
struct MaximumObservationHandler {
    float maximum_spot = -3.402823466e+38F;
    bool has_observation = false;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return on_observation(0U, state);
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        maximum_spot = fmaxf(maximum_spot, Dynamics::spot(state));
        has_observation = true;
        return true;
    }

    __device__ __forceinline__ float maximum() const {
        return has_observation ? maximum_spot : 0.0f;
    }
};

}  // namespace ai_factory::workbench::equity
