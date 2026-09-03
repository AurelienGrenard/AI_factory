// Compact SoA continuation-state projections for American equity options.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/handlers.cuh"
#include "common/longstaff_schwartz/basis/feature_vector.cuh"
#include "common/longstaff_schwartz/workspace.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::product {

template<equity::SpotDynamicsPolicy DynamicsPolicy>
struct SpotLogMoneynessContinuationState {
    using Dynamics = DynamicsPolicy;
    using RegressionInput =
        longstaff_schwartz::basis::TwoFactorInput;

    struct PreparedState {};

    struct StateView {
        float* spots;
    };

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    state_field_descriptors() {
        return {{sizeof(float), alignof(float)}};
    }

    static StateView make_state_view(
        unsigned char* workspace,
        const longstaff_schwartz::WorkspaceLayout& layout
    ) {
        if (layout.state_fields.size() != 1U) {
            throw std::logic_error(
                "Spot/log-moneyness continuation requires one state field."
            );
        }
        return {
            longstaff_schwartz::workspace_pointer<float>(
                workspace, layout.state_fields[0]
            ),
        };
    }

    __device__ __forceinline__ static PreparedState prepare(
        const typename Dynamics::Parameters&
    ) {
        return {};
    }

    __device__ __forceinline__ static auto make_writer(
        const StateView& states,
        std::size_t first_path_state,
        std::size_t paths_per_price,
        std::uint32_t write_count
    ) {
        return equity::SpotObservationWriter<Dynamics>{
            states.spots + first_path_state,
            paths_per_price,
            write_count,
        };
    }

    __device__ __forceinline__ static float spot(
        const StateView& states,
        std::size_t state_index
    ) {
        return states.spots[state_index];
    }

    __device__ __forceinline__ static RegressionInput regression_input(
        const PreparedState&,
        const StateView& states,
        std::size_t state_index,
        float inverse_strike
    ) {
        const float normalized_spot =
            states.spots[state_index] * inverse_strike;
        return {
            normalized_spot,
            logf(fmaxf(normalized_spot, 1.0e-6f)),
        };
    }
};

template<
    equity::SpotDynamicsPolicy DynamicsPolicy,
    auto StateMember,
    auto ScaleMember
>
struct SpotAndScaledStateContinuationState {
    using Dynamics = DynamicsPolicy;
    using RegressionInput =
        longstaff_schwartz::basis::TwoFactorInput;

    struct PreparedState {
        float inverse_scale;
    };

    struct StateView {
        float* spots;
        float* secondary_states;
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
                "Spot/scaled-state continuation requires two state fields."
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

    __device__ __forceinline__ static PreparedState prepare(
        const typename Dynamics::Parameters& parameters
    ) {
        return {1.0f / static_cast<float>(parameters.*ScaleMember)};
    }

    __device__ __forceinline__ static auto make_writer(
        const StateView& states,
        std::size_t first_path_state,
        std::size_t paths_per_price,
        std::uint32_t write_count
    ) {
        return equity::SpotAndStateObservationWriter<Dynamics, StateMember>{
            states.spots + first_path_state,
            states.secondary_states + first_path_state,
            paths_per_price,
            write_count,
        };
    }

    __device__ __forceinline__ static float spot(
        const StateView& states,
        std::size_t state_index
    ) {
        return states.spots[state_index];
    }

    __device__ __forceinline__ static RegressionInput regression_input(
        const PreparedState& prepared,
        const StateView& states,
        std::size_t state_index,
        float inverse_strike
    ) {
        return {
            states.spots[state_index] * inverse_strike,
            states.secondary_states[state_index] * prepared.inverse_scale,
        };
    }
};

}  // namespace ai_factory::workbench::product
