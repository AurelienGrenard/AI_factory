// Reusable path-local accumulators evaluated at contractual observations.
#pragma once

#include "common/compensated_sum.cuh"
#include "common/equity/concepts.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::equity {

template<SpotStatePolicy Dynamics>
struct ArithmeticMeanObservationHandler {
    CompensatedFloatSum spot_sum;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        return on_observation(0U, state);
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        spot_sum.add(Dynamics::spot(state));
        return true;
    }

    __device__ __forceinline__ float arithmetic_mean(
        std::uint32_t observation_count
    ) const {
        return observation_count == 0U
            ? 0.0f
            : spot_sum.value() / static_cast<float>(observation_count);
    }
};

template<LogSpotStatePolicy Dynamics, bool NativeLogSpot>
struct GeometricMeanObservationHandlerImplementation;

template<LogSpotStatePolicy Dynamics>
struct GeometricMeanObservationHandlerImplementation<Dynamics, true> {
    CompensatedFloatSum log_spot_sum;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        log_spot_sum.add(Dynamics::log_spot(state));
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        log_spot_sum.add(Dynamics::log_spot(state));
        return true;
    }

    __device__ __forceinline__ float geometric_mean(
        std::uint32_t observation_count
    ) const {
        if (observation_count == 0U) return 0.0f;
        return expf(
            log_spot_sum.value() / static_cast<float>(observation_count)
        );
    }
};

template<LogSpotStatePolicy Dynamics>
struct GeometricMeanObservationHandlerImplementation<Dynamics, false> {
    CompensatedFloatSum log_spot_sum;
    bool hit_non_positive_spot = false;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State& state
    ) {
        log_spot_sum.add(Dynamics::log_spot(state));
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State& state
    ) {
        const float spot = Dynamics::spot(state);
        hit_non_positive_spot =
            hit_non_positive_spot || !(spot > 0.0f);
        if (!hit_non_positive_spot) {
            log_spot_sum.add(Dynamics::log_spot(state));
        }
        return true;
    }

    __device__ __forceinline__ float geometric_mean(
        std::uint32_t observation_count
    ) const {
        if (observation_count == 0U || hit_non_positive_spot) return 0.0f;
        return expf(
            log_spot_sum.value() / static_cast<float>(observation_count)
        );
    }
};

template<LogSpotStatePolicy Dynamics>
using GeometricMeanObservationHandler =
    GeometricMeanObservationHandlerImplementation<
        Dynamics,
        Dynamics::kNativeLogSpot
    >;

template<SpotStatePolicy Dynamics>
struct MaximumObservationHandler {
    float maximum_spot = -3.402823466e+38F;

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
        return true;
    }

    __device__ __forceinline__ float maximum() const {
        return maximum_spot;
    }
};

// Write selected contractual spots to a caller-owned strided output view.
template<SpotStatePolicy Dynamics>
struct SpotObservationWriter {
    float* spots;
    std::size_t observation_stride;
    std::uint32_t write_count;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        if (observation < write_count) {
            spots[static_cast<std::size_t>(observation)
                  * observation_stride] = Dynamics::spot(state);
        }
        return true;
    }
};

// Write spots together with one float member of the model state.
template<SpotStatePolicy Dynamics, auto StateMember>
struct SpotAndStateObservationWriter {
    float* spots;
    float* state_values;
    std::size_t observation_stride;
    std::uint32_t write_count;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        if (observation < write_count) {
            const std::size_t output =
                static_cast<std::size_t>(observation)
                    * observation_stride;
            spots[output] = Dynamics::spot(state);
            state_values[output] = state.*StateMember;
        }
        return true;
    }
};

}  // namespace ai_factory::workbench::equity
