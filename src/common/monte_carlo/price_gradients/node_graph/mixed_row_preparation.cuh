// Build row-local axial and corner nodes for selected mixed derivatives.
#pragma once

#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/monte_carlo/price_gradients/node_graph/capacity.cuh"
#include "common/monte_carlo/price_gradients/node_graph/node_indices.cuh"
#include "common/price_gradients/device_preparation.cuh"
#include "common/price_gradients/mixed_sensitivity_preparation.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil.cuh"
#include "common/price_gradients/sensitivity_graph_plan.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

namespace node_graph_detail {

__host__ __device__ constexpr std::uint32_t larger_step_count(
    std::uint32_t first,
    std::uint32_t second
) {
    return first < second ? second : first;
}

template<
    typename Preparation,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities>
__host__ __device__ inline bool prepare_mixed_sensitivity_row_from_central(
    const typename Preparation::Scenario& central,
    const pg::SensitivitySpec<typename Preparation::Parameter>* sensitivities,
    std::size_t sensitivity_count,
    pg::DeviceSensitivityGraph graph,
    pg::TimeConfiguration time,
    typename Preparation::Scenario* scenarios,
    pg::SensitivityStencil<4U>* stencils,
    SensitivityNodeIndices<4U>* axis_node_indices,
    pg::MixedSensitivityStencil* mixed_stencils,
    MixedSensitivityNodeIndices* mixed_node_indices,
    std::uint16_t& node_count,
    std::uint32_t& maximum_steps,
    int& error,
    std::size_t& error_sensitivity
) {
    constexpr auto node_capacity = mixed_node_graph_node_capacity<
        MaximumSensitivities, MaximumMixedSensitivities
    >();
    if (sensitivity_count == 0U
        || sensitivity_count > MaximumSensitivities
        || graph.coordinate_use_count != sensitivity_count
        || graph.coordinate_use_capacity < sensitivity_count
        || graph.mixed_second_count == 0U
        || graph.mixed_second_count > MaximumMixedSensitivities
        || graph.mixed_second_capacity < graph.mixed_second_count
        || graph.node_capacity > node_capacity) {
        error = preparation::unsupported_order;
        return false;
    }

    scenarios[0U] = central;
    node_count = 1U;
    maximum_steps = central.step_count;
    for (std::size_t sensitivity = 0U;
         sensitivity < sensitivity_count;
         ++sensitivity) {
        auto& indices = axis_node_indices[sensitivity];
        indices[0U] = 0U;
        const auto use = graph.coordinate_uses[sensitivity];
        if (use == pg::SensitivityCoordinateUse::none) {
            stencils[sensitivity] = {};
            continue;
        }

        typename Preparation::Scenario nodes[4U]{};
        if (!pg::prepare_axis_sensitivity_nodes<Preparation>(
                central,
                sensitivities[sensitivity],
                use,
                time,
                stencils[sensitivity],
                nodes,
                error
            )) {
            error_sensitivity = sensitivity;
            return false;
        }

        const auto active = pg::active_node_count(stencils[sensitivity]);
        for (std::size_t local_node = 1U;
             local_node < active;
             ++local_node) {
            if (node_count >= graph.node_capacity
                || node_count >= node_capacity) {
                error = preparation::unsupported_order;
                error_sensitivity = sensitivity;
                return false;
            }
            indices[local_node] = node_count;
            scenarios[node_count] = nodes[local_node];
            maximum_steps = larger_step_count(
                maximum_steps, scenarios[node_count].step_count
            );
            ++node_count;
        }
    }

    for (std::size_t pair_index = 0U;
         pair_index < graph.mixed_second_count;
         ++pair_index) {
        const auto pair = graph.mixed_second[pair_index];
        if (pair.first >= sensitivity_count
            || pair.second >= sensitivity_count
            || pair.first >= pair.second) {
            error = preparation::unsupported_order;
            error_sensitivity = pair.first;
            return false;
        }
        const auto mixed = pg::make_mixed_sensitivity_stencil(
            stencils[pair.first], stencils[pair.second]
        );
        mixed_stencils[pair_index] = mixed;
        auto& indices = mixed_node_indices[pair_index];
        std::uint16_t corners[4U]{0xffffU, 0xffffU, 0xffffU, 0xffffU};
        for (std::size_t local = 0U; local < mixed.node_count; ++local) {
            const auto first_local = mixed.first_local_nodes[local];
            const auto second_local = mixed.second_local_nodes[local];
            if (first_local == 0U && second_local == 0U) {
                indices[local] = 0U;
                continue;
            }
            if (first_local == 0U) {
                indices[local] = axis_node_indices[pair.second][second_local];
                continue;
            }
            if (second_local == 0U) {
                indices[local] = axis_node_indices[pair.first][first_local];
                continue;
            }

            const auto corner = static_cast<std::size_t>(first_local - 1U)
                * 2U + static_cast<std::size_t>(second_local - 1U);
            if (corners[corner] == 0xffffU) {
                if (node_count >= graph.node_capacity
                    || node_count >= node_capacity) {
                    error = preparation::unsupported_order;
                    error_sensitivity = pair.first;
                    return false;
                }
                auto& scenario = scenarios[node_count];
                if (!pg::prepare_mixed_corner_scenario<Preparation>(
                        central,
                        sensitivities[pair.first],
                        stencils[pair.first].parameter_values[first_local],
                        scenarios[
                            axis_node_indices[pair.first][first_local]
                        ],
                        sensitivities[pair.second],
                        stencils[pair.second].parameter_values[second_local],
                        scenarios[
                            axis_node_indices[pair.second][second_local]
                        ],
                        time,
                        scenario
                    )) {
                    error = preparation::no_admissible_stencil;
                    error_sensitivity = pair.first;
                    return false;
                }
                maximum_steps = larger_step_count(
                    maximum_steps, scenario.step_count
                );
                corners[corner] = node_count++;
            }
            indices[local] = corners[corner];
        }
    }
    return true;
}

template<
    typename Inputs,
    typename Preparation,
    std::size_t MaximumSensitivities,
    std::size_t MaximumMixedSensitivities>
__device__ __forceinline__ bool prepare_mixed_sensitivity_row(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::DeviceSensitivityGraph graph,
    std::size_t row,
    typename Preparation::Scenario* scenarios,
    pg::SensitivityStencil<4U>* stencils,
    SensitivityNodeIndices<4U>* axis_node_indices,
    pg::MixedSensitivityStencil* mixed_stencils,
    MixedSensitivityNodeIndices* mixed_node_indices,
    std::uint16_t& node_count,
    std::uint32_t& maximum_steps,
    int& error,
    std::size_t& error_sensitivity
) {
    typename Preparation::Scenario central{};
    if (!inputs.make_central(row, plan, central)) {
        error = preparation::invalid_central;
        error_sensitivity = 0U;
        return false;
    }
    return prepare_mixed_sensitivity_row_from_central<
        Preparation, MaximumSensitivities, MaximumMixedSensitivities
    >(
        central,
        inputs.sensitivities,
        plan.sensitivity_count,
        graph,
        plan.time,
        scenarios,
        stencils,
        axis_node_indices,
        mixed_stencils,
        mixed_node_indices,
        node_count,
        maximum_steps,
        error,
        error_sensitivity
    );
}

}  // namespace node_graph_detail

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
