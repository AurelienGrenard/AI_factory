// One block per (row, sensitivity), with finite-difference nodes built on device.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/philox.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/row_mapping.cuh"
#include "common/monte_carlo/price_gradients/coupled_terminal_simulation.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_stencil_kernel.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::equity::price_gradients::device_preparation;

namespace detail {

template<typename Dynamics, typename ProductPolicy, typename Preparation>
struct FirstTerminalSensitivityPolicy {
    using NodePolicy =
        SelectedTerminalNodePolicy<Dynamics, ProductPolicy, Preparation>;
    using Scenario = typename Preparation::Scenario;
    using Stencil = pg::SensitivityStencil<3U>;
    using Nodes = pg::SensitivityNodes<Scenario, 3U>;
    using NodeValue = typename NodePolicy::NodeValue;

    struct PreparedRow {
        typename Dynamics::Prepared dynamics[3U];
        typename NodePolicy::Metadata metadata[3U];
        const Nodes* nodes;
        const Stencil* stencil;
        std::uint32_t maximum_steps;

        __device__ __forceinline__ const Scenario& input(
            unsigned int node
        ) const {
            return (*nodes)[node];
        }
    };

    template<bool IncludeCentral>
    __device__ __forceinline__ static NodeValue terminal_value(
        const PreparedRow& row,
        const typename Dynamics::State* states,
        unsigned int node
    ) {
        if constexpr (IncludeCentral) {
            return NodePolicy::observe(
                row.input(node).reuse_central ? states[0U] : states[node]
            );
        } else {
            return NodePolicy::observe(states[node - 1U]);
        }
    }

    template<bool IncludeCentral>
    __device__ __forceinline__ static float finalize(
        const PreparedRow& row,
        const typename Dynamics::State* states,
        unsigned int node
    ) {
        return NodePolicy::payoff(
            row.metadata[node],
            terminal_value<IncludeCentral>(row, states, node)
        );
    }

    __device__ __forceinline__ static PreparedRow prepare(
        const Nodes* nodes,
        const Stencil* stencil,
        pg::TimeConfiguration time,
        bool include_central
    ) {
        PreparedRow row{};
        row.nodes = nodes;
        row.stencil = stencil;
        #pragma unroll
        for (unsigned int node = 0U; node < 3U; ++node) {
            const auto& input = row.input(node);
            if (node == 0U && !include_central) {
                if constexpr (Dynamics::kDrawRequiresCentralPrepared) {
                    row.dynamics[0U] =
                        NodePolicy::prepare_dynamics(input, time);
                }
                continue;
            }
            row.dynamics[node] =
                NodePolicy::prepare_dynamics(input, time);
            row.metadata[node] =
                NodePolicy::prepare_metadata(input, time);
            row.maximum_steps = max(row.maximum_steps, input.step_count);
        }
        return row;
    }

    struct PathResult {
        float price;
        float sensitivity;
    };

    template<bool IncludeCentral, bool IncludePayoff>
    __device__ __forceinline__ static PathResult evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        static_assert(IncludeCentral || !IncludePayoff);
        constexpr unsigned int first_node = IncludeCentral ? 0U : 1U;
        typename Dynamics::State states[3U - first_node];
        #pragma unroll
        for (unsigned int node = first_node; node < 3U; ++node) {
            states[node - first_node] = Dynamics::initial(row.dynamics[node]);
        }
        typename Dynamics::RandomContext random(key, path);
        simulate_coupled_terminal_nodes<3U, Dynamics>(
            random,
            row.dynamics,
            3U,
            row.maximum_steps,
            [&](unsigned int node) {
                if (node == 0U) return IncludeCentral;
                return !row.input(node).reuse_central;
            },
            [&](unsigned int node) { return row.input(node).step_count; },
            [&](unsigned int node) {
                return row.input(node).normal_weights;
            },
            [&](unsigned int node) -> typename Dynamics::State& {
                return states[node - first_node];
            }
        );
        PathResult result{};
        if constexpr (IncludePayoff) {
            result.price = finalize<IncludeCentral>(row, states, 0U);
        }
        if (row.stencil->kind == pg::StencilKind::centered) {
            result.sensitivity = NodePolicy::centered_first(
                row.metadata[1U],
                row.metadata[2U],
                terminal_value<IncludeCentral>(row, states, 1U),
                terminal_value<IncludeCentral>(row, states, 2U),
                row.stencil->represented_width
            );
        } else if constexpr (IncludePayoff) {
            pg::SensitivityValues<3U> values{};
            values[0U] = result.price;
            values[1U] = finalize<IncludeCentral>(row, states, 1U);
            values[2U] = finalize<IncludeCentral>(row, states, 2U);
            result.sensitivity = pg::reconstruct_sensitivity<
                pg::SensitivityOrders::first
            >(*row.stencil, values).first;
        }
        return result;
    }
};

}  // namespace detail
}  // namespace ai_factory::workbench::monte_carlo::price_gradients

#include "common/monte_carlo/price_gradients/device_prepared_first_sensitivity_kernel.cuh"

namespace ai_factory::workbench::monte_carlo::price_gradients {

template<typename Dynamics, typename ProductPolicy, typename Preparation, typename Inputs>
void launch_device_prepared_terminal_first_sensitivities(
    Inputs inputs,
    DevicePreparedPlan plan,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs,
    DevicePreparedStencilOutputs<3U> stencil_outputs,
    const char* kernel_name,
    const char* variant
) {
    using Policy = detail::FirstTerminalSensitivityPolicy<
        Dynamics, ProductPolicy, Preparation
    >;
    launch_device_prepared_first_sensitivities<Policy, Preparation>(
        inputs,
        plan,
        configuration,
        outputs,
        stencil_outputs,
        kernel_name,
        variant
    );
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
