// Reusable model-state projections and coalesced sample output writers.
#pragma once

#include "common/check_cuda.cuh"
#include "common/equity/concepts.cuh"
#include "common/simulation/concepts.cuh"

#include <cuda_runtime.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::sample {

struct ScalarSampleOutput {
    float* values;
};

struct PairSampleOutput {
    float* first_values;
    float* second_values;
};

template<typename Dynamics>
requires std::convertible_to<typename Dynamics::State, float>
struct StateValueAccessor {
    __device__ __forceinline__ static float value(
        const typename Dynamics::State& state
    ) {
        return static_cast<float>(state);
    }
};

template<typename Dynamics>
requires equity::SpotStatePolicy<Dynamics>
struct SpotValueAccessor {
    __device__ __forceinline__ static float value(
        const typename Dynamics::State& state
    ) {
        return Dynamics::spot(state);
    }
};

template<typename Dynamics, auto StateMember>
struct StateMemberValueAccessor {
    __device__ __forceinline__ static float value(
        const typename Dynamics::State& state
    ) {
        return static_cast<float>(state.*StateMember);
    }
};

template<typename Accessor, typename Dynamics>
concept SampleValueAccessorFor = requires(
    const typename Dynamics::State& state
) {
    { Accessor::value(state) } -> std::same_as<float>;
};

template<
    simulation::StatePolicy Dynamics,
    typename Accessor
>
requires SampleValueAccessorFor<Accessor, Dynamics>
struct ScalarSampleObservation {
    using Output = ScalarSampleOutput;

    struct Handler {
        float* values;
        std::size_t observation_stride;

        __device__ __forceinline__ bool on_initial_state(
            const typename Dynamics::State&
        ) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t observation,
            const typename Dynamics::State& state
        ) {
            values[
                static_cast<std::size_t>(observation) * observation_stride
            ] = Accessor::value(state);
            return true;
        }
    };

    static void validate(const Output& output) {
        validate_device_pointer(output.values, "device sample values");
    }

    __device__ __forceinline__ static void write_terminal(
        const Output& output,
        std::size_t sample_index,
        const typename Dynamics::State& state
    ) {
        output.values[sample_index] = Accessor::value(state);
    }

    __device__ __forceinline__ static Handler make_handler(
        const Output& output,
        std::size_t sample_index,
        std::size_t total_sample_count
    ) {
        return {
            output.values + sample_index,
            total_sample_count,
        };
    }
};

template<
    simulation::StatePolicy Dynamics,
    typename FirstAccessor,
    typename SecondAccessor
>
requires (
    SampleValueAccessorFor<FirstAccessor, Dynamics>
    && SampleValueAccessorFor<SecondAccessor, Dynamics>
)
struct PairSampleObservation {
    using Output = PairSampleOutput;

    struct Handler {
        float* first_values;
        float* second_values;
        std::size_t observation_stride;

        __device__ __forceinline__ bool on_initial_state(
            const typename Dynamics::State&
        ) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t observation,
            const typename Dynamics::State& state
        ) {
            const std::size_t output =
                static_cast<std::size_t>(observation) * observation_stride;
            first_values[output] = FirstAccessor::value(state);
            second_values[output] = SecondAccessor::value(state);
            return true;
        }
    };

    static void validate(const Output& output) {
        validate_device_pointer(
            output.first_values,
            "device primary sample values"
        );
        validate_device_pointer(
            output.second_values,
            "device secondary sample values"
        );
    }

    __device__ __forceinline__ static void write_terminal(
        const Output& output,
        std::size_t sample_index,
        const typename Dynamics::State& state
    ) {
        output.first_values[sample_index] = FirstAccessor::value(state);
        output.second_values[sample_index] = SecondAccessor::value(state);
    }

    __device__ __forceinline__ static Handler make_handler(
        const Output& output,
        std::size_t sample_index,
        std::size_t total_sample_count
    ) {
        return {
            output.first_values + sample_index,
            output.second_values + sample_index,
            total_sample_count,
        };
    }
};

template<typename Dynamics>
using StateSampleObservation = ScalarSampleObservation<
    Dynamics,
    StateValueAccessor<Dynamics>
>;

template<typename Dynamics>
using SpotSampleObservation = ScalarSampleObservation<
    Dynamics,
    SpotValueAccessor<Dynamics>
>;

template<typename Dynamics, auto StateMember>
using StateMemberSampleObservation = ScalarSampleObservation<
    Dynamics,
    StateMemberValueAccessor<Dynamics, StateMember>
>;

template<typename Dynamics, auto StateMember>
using SpotAndStateSampleObservation = PairSampleObservation<
    Dynamics,
    SpotValueAccessor<Dynamics>,
    StateMemberValueAccessor<Dynamics, StateMember>
>;

template<typename Dynamics, auto FirstMember, auto SecondMember>
using TwoStateMemberSampleObservation = PairSampleObservation<
    Dynamics,
    StateMemberValueAccessor<Dynamics, FirstMember>,
    StateMemberValueAccessor<Dynamics, SecondMember>
>;

}  // namespace ai_factory::workbench::sample
