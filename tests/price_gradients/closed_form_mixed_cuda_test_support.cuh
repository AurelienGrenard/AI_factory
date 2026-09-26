// Shared public-API fixture for selected mixed closed-form derivatives.
#pragma once

#include "diagonal_cuda_test_support.cuh"
#include "common/price_gradients/mixed_sensitivity_outputs.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace price_gradient_test {

struct ClosedFormMixedResults {
    std::vector<float> prices;
    std::vector<float> gradients;
    std::vector<float> diagonal_hessians;
    std::vector<float> mixed_hessians;
    std::vector<pg::SensitivityStencil<4U>> stencils;
    std::vector<pg::MixedSensitivityStencil> mixed_stencils;
};

template<typename Plan, typename WorkspaceSizer, typename Launcher>
ClosedFormMixedResults execute_closed_form_mixed(
    const Plan& plan,
    pg::LaunchConfiguration launch,
    WorkspaceSizer workspace_size,
    Launcher launcher,
    const char* label
) {
    const auto rows = plan.result_count;
    const auto sensitivity_count = plan.sensitivity_count();
    const auto first_count = plan.sensitivity_graph.first.size();
    const auto diagonal_count =
        plan.sensitivity_graph.diagonal_second.size();
    const auto selected_capacity = rows * std::max(
        first_count, diagonal_count
    );
    const auto mixed_count = plan.sensitivity_graph.mixed_second.size();
    launch.method = pg::PricingMethod::closed_form;
    launch.result_offset = 0U;
    launch.result_count = rows;

    DeviceInputStorage<Plan> input_storage(plan);
    DeviceArray<pg::SensitivityStencil<4U>> stencils(
        rows * sensitivity_count
    );
    DeviceArray<pg::MixedSensitivityStencil> mixed_stencils(
        rows * mixed_count
    );
    DeviceArray<pg::device_preparation::Error> preparation_error(1U);
    DeviceArray<std::uint8_t> workspace(workspace_size(plan, launch));
    DeviceArray<float> prices(rows);
    DeviceArray<float> gradients(selected_capacity);
    DeviceArray<float> diagonal_hessians(selected_capacity);
    DeviceArray<float> mixed_hessians(rows * mixed_count);

    const typename Plan::DiagonalStencilOutputs stencil_outputs{
        stencils.data, stencils.count, preparation_error.data
    };
    const typename Plan::MixedStencilOutputs mixed_stencil_outputs{
        mixed_stencils.data, mixed_stencils.count
    };
    const pg::SensitivityOutputs outputs{
        prices.data,
        nullptr,
        gradients.data,
        nullptr,
        diagonal_hessians.data,
        nullptr,
        rows,
        selected_capacity,
    };
    const pg::MixedSensitivityOutputs mixed_outputs{
        mixed_hessians.data,
        nullptr,
        rows * mixed_count,
    };

    launcher(
        plan,
        input_storage.inputs(),
        stencil_outputs,
        mixed_stencil_outputs,
        launch,
        outputs,
        mixed_outputs,
        workspace.data,
        workspace.count
    );
    check_cuda(cudaDeviceSynchronize(), label);
    const auto error = preparation_error.read()[0U];
    if (error.code != 0) {
        throw std::runtime_error(
            std::string(label) + " preparation failed at row "
            + std::to_string(error.row)
            + ", sensitivity " + std::to_string(error.sensitivity)
            + ", code " + std::to_string(error.code)
        );
    }

    const auto mixed = mixed_hessians.read();
    for (const auto value : mixed) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                std::string(label) + " mixed Hessian is not finite."
            );
        }
    }
    return {
        prices.read(),
        gradients.read(),
        diagonal_hessians.read(),
        mixed,
        stencils.read(),
        mixed_stencils.read(),
    };
}

inline void require_closed_form_diagonal_parity(
    const DiagonalResults& expected,
    const ClosedFormMixedResults& actual,
    const std::string& label
) {
    require_same_stencils(expected.stencils, actual.stencils, label);
    require_same_bits(expected.price, actual.prices, label + " price");
    require_same_bits(
        expected.gradient, actual.gradients, label + " gradient"
    );
    require_same_bits(
        expected.diagonal_hessian,
        actual.diagonal_hessians,
        label + " diagonal Hessian"
    );
}

}  // namespace price_gradient_test
