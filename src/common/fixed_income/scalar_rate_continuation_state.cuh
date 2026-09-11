// Minimal SoA scalar-rate projection for pricing without a pathwise integral.
#pragma once

#include "common/longstaff_schwartz/workspace.cuh"
#include <stdexcept>

namespace ai_factory::workbench::fixed_income {

template<typename DynamicsPolicy>
struct ScalarRateContinuationState {
    using Dynamics = DynamicsPolicy;
    using RegressionInput = float;
    struct StateView { float* factors; };
    struct Writer {
        StateView states;
        std::size_t offset;
        std::size_t stride;
        std::uint32_t write_count;
        __device__ __forceinline__ bool on_initial_state(float) { return true; }
        __device__ __forceinline__ bool on_observation(std::uint32_t exercise, float state) {
            if (exercise < write_count)
                states.factors[offset + static_cast<std::size_t>(exercise) * stride] = state;
            return true;
        }
    };
    static std::vector<longstaff_schwartz::StateFieldDescriptor> state_field_descriptors() {
        return {{sizeof(float), alignof(float)}};
    }
    static StateView make_state_view(
        unsigned char* workspace, const longstaff_schwartz::WorkspaceLayout& layout
    ) {
        if (layout.state_fields.size() != 1U)
            throw std::logic_error("A scalar rate continuation requires exactly one field.");
        return {longstaff_schwartz::workspace_pointer<float>(workspace, layout.state_fields[0])};
    }
    __device__ __forceinline__ static Writer make_writer(
        StateView states, std::size_t offset, std::size_t stride, std::uint32_t count
    ) { return {states, offset, stride, count}; }
    __device__ __forceinline__ static float factor(StateView states, std::size_t index) {
        return states.factors[index];
    }
    template<typename Analytics>
    __device__ __forceinline__ static float regression_input(
        const typename Analytics::PreparedRegressionState& prepared,
        StateView states, std::size_t index
    ) { return Analytics::normalize_regression_state(prepared, factor(states, index)); }
};

}  // namespace ai_factory::workbench::fixed_income
