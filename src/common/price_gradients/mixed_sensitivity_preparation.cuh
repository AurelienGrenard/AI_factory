// Engine-neutral preparation of axial and corner finite-difference nodes.
#pragma once

#include "common/price_gradients/device_preparation.cuh"
#include "common/price_gradients/sensitivity_graph_plan.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::price_gradients {

namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

template<typename Scenario>
__host__ __device__ inline void copy_first_sensitivity_task(
    const SensitivityTask<Scenario, 3U>& source,
    SensitivityStencil<4U>& stencil,
    Scenario (&nodes)[4U]
) {
    stencil = promote_first_sensitivity_stencil(source.stencil);
    for (std::size_t node = 0U; node < 3U; ++node) {
        nodes[node] = source.nodes[node];
    }
}

template<typename Preparation>
__host__ __device__ inline bool prepare_axis_sensitivity_nodes(
    const typename Preparation::Scenario& central,
    SensitivitySpec<typename Preparation::Parameter> sensitivity,
    SensitivityCoordinateUse use,
    TimeConfiguration time,
    SensitivityStencil<4U>& stencil,
    typename Preparation::Scenario (&nodes)[4U],
    int& error
) {
    if (has_coordinate_use(
            use, SensitivityCoordinateUse::diagonal_second
        )) {
        SensitivityTask<typename Preparation::Scenario, 4U> task{};
        if (!preparation::build_sensitivity_task<
                SensitivityOrders::first_and_second,
                Preparation
            >(central, sensitivity, time, task, error)) {
            return false;
        }
        stencil = task.stencil;
        for (std::size_t node = 0U;
             node < active_node_count(task.stencil);
             ++node) {
            nodes[node] = task.nodes[node];
        }
        return true;
    }

    SensitivityTask<typename Preparation::Scenario, 3U> task{};
    if (!preparation::build_sensitivity_task<
            SensitivityOrders::first,
            Preparation
        >(central, sensitivity, time, task, error)) {
        return false;
    }
    copy_first_sensitivity_task(task, stencil, nodes);
    return true;
}

template<typename Preparation>
__host__ __device__ inline bool prepare_mixed_corner_scenario(
    const typename Preparation::Scenario& central,
    SensitivitySpec<typename Preparation::Parameter> first,
    float first_value,
    const typename Preparation::Scenario& first_axis_node,
    SensitivitySpec<typename Preparation::Parameter> second,
    float second_value,
    const typename Preparation::Scenario& second_axis_node,
    TimeConfiguration time,
    typename Preparation::Scenario& corner
) {
    if (!Preparation::change_scenario_pair(
            central,
            first.parameter,
            first_value,
            second.parameter,
            second_value,
            time,
            corner
        )) {
        return false;
    }
    Preparation::finalize_mixed_scenario(
        first.parameter,
        first_axis_node,
        second.parameter,
        second_axis_node,
        corner
    );
    return true;
}

template<typename Preparation>
__host__ __device__ inline bool prepare_mixed_corner_from_stencils(
    const typename Preparation::Scenario& central,
    SensitivitySpec<typename Preparation::Parameter> first,
    const SensitivityStencil<4U>& first_stencil,
    std::size_t first_local_node,
    SensitivitySpec<typename Preparation::Parameter> second,
    const SensitivityStencil<4U>& second_stencil,
    std::size_t second_local_node,
    TimeConfiguration time,
    typename Preparation::Scenario& corner
) {
    typename Preparation::Scenario first_axis_node{};
    typename Preparation::Scenario second_axis_node{};
    if (!Preparation::change_scenario(
            central,
            first.parameter,
            first_stencil.parameter_values[first_local_node],
            time,
            first_axis_node
        )
        || !Preparation::change_scenario(
            central,
            second.parameter,
            second_stencil.parameter_values[second_local_node],
            time,
            second_axis_node
        )) {
        return false;
    }
    return prepare_mixed_corner_scenario<Preparation>(
        central,
        first,
        first_stencil.parameter_values[first_local_node],
        first_axis_node,
        second,
        second_stencil.parameter_values[second_local_node],
        second_axis_node,
        time,
        corner
    );
}

}  // namespace ai_factory::workbench::price_gradients
