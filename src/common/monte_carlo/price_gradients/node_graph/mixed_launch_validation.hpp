// Engine-neutral validation for selected mixed sensitivity node graphs.
#pragma once

#include "common/price_gradients/device_prepared_validation.hpp"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil_outputs.cuh"
#include "common/price_gradients/sensitivity_graph_plan.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::monte_carlo::price_gradients {

inline void validate_device_sensitivity_graph(
    pg::DeviceSensitivityGraph device,
    const pg::SensitivityGraphPlan& host,
    std::size_t sensitivity_count
) {
    const bool shape_mismatch =
        device.first_count != host.first.size()
        || device.diagonal_second_count != host.diagonal_second.size()
        || device.mixed_second_count != host.mixed_second.size()
        || device.coordinate_use_count != host.coordinate_uses.size()
        || device.node_capacity != host.node_capacity
        || device.coordinate_use_count != sensitivity_count;
    if (shape_mismatch
        || device.first_capacity < device.first_count
        || device.diagonal_second_capacity < device.diagonal_second_count
        || device.mixed_second_capacity < device.mixed_second_count
        || device.coordinate_use_capacity < device.coordinate_use_count) {
        throw std::invalid_argument(
            "Mixed device graph does not match its host plan."
        );
    }
    const auto validate = [](const void* pointer,
                             std::size_t count,
                             const char* label) {
        if (count != 0U) validate_device_pointer(pointer, label);
    };
    validate(device.first, device.first_count, "mixed graph first indices");
    validate(
        device.diagonal_second,
        device.diagonal_second_count,
        "mixed graph diagonal indices"
    );
    validate(
        device.mixed_second,
        device.mixed_second_count,
        "mixed graph pairs"
    );
    validate(
        device.coordinate_uses,
        device.coordinate_use_count,
        "mixed graph coordinate uses"
    );
}

inline void validate_mixed_output_views(
    std::size_t rows,
    std::size_t first_count,
    std::size_t diagonal_count,
    std::size_t mixed_count,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs
) {
    if (outputs.prices == nullptr
        || outputs.price_standard_errors == nullptr
        || outputs.price_capacity < rows
        || (first_count != 0U
            && (outputs.gradients == nullptr
                || outputs.gradient_standard_errors == nullptr
                || outputs.sensitivity_capacity < rows * first_count))
        || (diagonal_count != 0U
            && (outputs.diagonal_hessians == nullptr
                || outputs.diagonal_hessian_standard_errors == nullptr
                || outputs.sensitivity_capacity < rows * diagonal_count))
        || (mixed_count != 0U
            && (mixed_outputs.hessians == nullptr
                || mixed_outputs.standard_errors == nullptr
                || mixed_outputs.capacity < rows * mixed_count
                || mixed_stencil_outputs.stencils == nullptr
                || mixed_stencil_outputs.capacity < rows * mixed_count))) {
        throw std::invalid_argument(
            "Insufficient mixed numerical output capacity."
        );
    }
}

template<typename Inputs>
void validate_mixed_node_graph_launch(
    Inputs inputs,
    DevicePreparedPlan plan,
    const pg::SensitivityGraphPlan& host_graph,
    pg::DeviceSensitivityGraph device_graph,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    pg::MixedSensitivityOutputs mixed_outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    pg::MixedSensitivityStencilOutputs mixed_stencil_outputs
) {
    validate_device_sensitivity_graph(
        device_graph, host_graph, plan.sensitivity_count
    );
    const auto rows = plan.result_count;
    const auto axis_stencil_count = rows * plan.sensitivity_count;
    const auto mixed_stencil_count =
        rows * host_graph.mixed_second.size();
    if (stencil_outputs.stencils == nullptr
        || stencil_outputs.capacity < axis_stencil_count
        || stencil_outputs.error == nullptr) {
        throw std::invalid_argument(
            "Insufficient mixed represented-stencil output capacity."
        );
    }
    validate_mixed_output_views(
        rows,
        host_graph.first.size(),
        host_graph.diagonal_second.size(),
        host_graph.mixed_second.size(),
        outputs,
        mixed_outputs,
        mixed_stencil_outputs
    );

    std::vector<pg::BufferRange> numerical_outputs{
        pg::checked_buffer_range(outputs.prices, rows, sizeof(float)),
        pg::checked_buffer_range(
            outputs.price_standard_errors, rows, sizeof(float)
        ),
        pg::checked_buffer_range(
            mixed_outputs.hessians, mixed_stencil_count, sizeof(float)
        ),
        pg::checked_buffer_range(
            mixed_outputs.standard_errors, mixed_stencil_count, sizeof(float)
        ),
        pg::checked_buffer_range(
            mixed_stencil_outputs.stencils,
            mixed_stencil_count,
            sizeof(pg::MixedSensitivityStencil)
        ),
    };
    if (!host_graph.first.empty()) {
        numerical_outputs.push_back(pg::checked_buffer_range(
            outputs.gradients, rows * host_graph.first.size(), sizeof(float)
        ));
        numerical_outputs.push_back(pg::checked_buffer_range(
            outputs.gradient_standard_errors,
            rows * host_graph.first.size(),
            sizeof(float)
        ));
    }
    if (!host_graph.diagonal_second.empty()) {
        numerical_outputs.push_back(pg::checked_buffer_range(
            outputs.diagonal_hessians,
            rows * host_graph.diagonal_second.size(),
            sizeof(float)
        ));
        numerical_outputs.push_back(pg::checked_buffer_range(
            outputs.diagonal_hessian_standard_errors,
            rows * host_graph.diagonal_second.size(),
            sizeof(float)
        ));
    }
    validate_device_prepared_launch(
        inputs,
        plan,
        stencil_outputs,
        launch,
        numerical_outputs
    );
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
