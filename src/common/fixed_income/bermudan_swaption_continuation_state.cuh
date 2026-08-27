// Generic SoA projections of joint short-rate states for early exercise.
#pragma once

#include "common/longstaff_schwartz/basis/feature_vector.cuh"
#include "common/longstaff_schwartz/workspace.cuh"
#include "common/simulation/concepts.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ai_factory::workbench::fixed_income {

template<simulation::DynamicsPolicy DynamicsPolicy>
struct OneFactorRateContinuationState {
    using Dynamics = DynamicsPolicy;
    using RegressionInput = float;

    struct StateView {
        float* factors;
        float* integrals;
    };

    struct Writer {
        StateView states;
        std::size_t first_path_state;
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
                const std::size_t output = first_path_state
                    + static_cast<std::size_t>(observation)
                        * observation_stride;
                states.factors[output] = state.state;
                states.integrals[output] = state.state_integral;
            }
            return true;
        }
    };

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    state_field_descriptors() {
        return {
            {sizeof(float), alignof(float)},
            {sizeof(float), alignof(float)},
        };
    }

    static StateView make_state_view(
        unsigned char* workspace,
        const longstaff_schwartz::WorkspaceLayout& layout
    ) {
        if (layout.state_fields.size() != 2U) {
            throw std::logic_error(
                "A one-factor rate continuation requires factor and integral fields."
            );
        }
        return {
            longstaff_schwartz::workspace_pointer<float>(
                workspace, layout.state_fields[0]
            ),
            longstaff_schwartz::workspace_pointer<float>(
                workspace, layout.state_fields[1]
            ),
        };
    }

    __device__ __forceinline__ static Writer make_writer(
        const StateView& states,
        std::size_t first_path_state,
        std::size_t paths_per_price,
        std::uint32_t write_count
    ) {
        return {states, first_path_state, paths_per_price, write_count};
    }

    __device__ __forceinline__ static float factor(
        const StateView& states,
        std::size_t state_index
    ) {
        return states.factors[state_index];
    }

    __device__ __forceinline__ static float integral(
        const StateView& states,
        std::size_t state_index
    ) {
        return states.integrals[state_index];
    }

    template<typename RegressionProjection>
    __device__ __forceinline__ static RegressionInput regression_input(
        const typename RegressionProjection::PreparedRegressionState&
            prepared,
        const StateView& states,
        std::size_t state_index
    ) {
        return RegressionProjection::normalize_regression_state(
            prepared, states.factors[state_index]
        );
    }
};

template<simulation::DynamicsPolicy DynamicsPolicy>
struct TwoFactorRateContinuationState {
    using Dynamics = DynamicsPolicy;
    using RegressionInput = longstaff_schwartz::basis::TwoFactorInput;
    using FactorState = std::remove_cvref_t<decltype(
        std::declval<typename Dynamics::State>().state
    )>;

    struct StateView {
        float* factors_x;
        float* factors_y;
        float* integrals;
    };

    struct Writer {
        StateView states;
        std::size_t first_path_state;
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
                const std::size_t output = first_path_state
                    + static_cast<std::size_t>(observation)
                        * observation_stride;
                states.factors_x[output] = state.state.state_x;
                states.factors_y[output] = state.state.state_y;
                states.integrals[output] = state.state_integral;
            }
            return true;
        }
    };

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    state_field_descriptors() {
        return {
            {sizeof(float), alignof(float)},
            {sizeof(float), alignof(float)},
            {sizeof(float), alignof(float)},
        };
    }

    static StateView make_state_view(
        unsigned char* workspace,
        const longstaff_schwartz::WorkspaceLayout& layout
    ) {
        if (layout.state_fields.size() != 3U) {
            throw std::logic_error(
                "A two-factor rate continuation requires two factor fields and one integral field."
            );
        }
        return {
            longstaff_schwartz::workspace_pointer<float>(
                workspace, layout.state_fields[0]
            ),
            longstaff_schwartz::workspace_pointer<float>(
                workspace, layout.state_fields[1]
            ),
            longstaff_schwartz::workspace_pointer<float>(
                workspace, layout.state_fields[2]
            ),
        };
    }

    __device__ __forceinline__ static Writer make_writer(
        const StateView& states,
        std::size_t first_path_state,
        std::size_t paths_per_price,
        std::uint32_t write_count
    ) {
        return {states, first_path_state, paths_per_price, write_count};
    }

    __device__ __forceinline__ static FactorState factor(
        const StateView& states,
        std::size_t state_index
    ) {
        return FactorState{
            states.factors_x[state_index],
            states.factors_y[state_index],
        };
    }

    __device__ __forceinline__ static float integral(
        const StateView& states,
        std::size_t state_index
    ) {
        return states.integrals[state_index];
    }

    template<typename RegressionProjection>
    __device__ __forceinline__ static RegressionInput regression_input(
        const typename RegressionProjection::PreparedRegressionState&
            prepared,
        const StateView& states,
        std::size_t state_index
    ) {
        return {
            RegressionProjection::normalize_regression_state_x(
                prepared, states.factors_x[state_index]
            ),
            RegressionProjection::normalize_regression_state_y(
                prepared, states.factors_y[state_index]
            ),
        };
    }
};

}  // namespace ai_factory::workbench::fixed_income
