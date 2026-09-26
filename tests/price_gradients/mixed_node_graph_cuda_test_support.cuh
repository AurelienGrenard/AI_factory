// Shared public-API fixture for selected mixed-Hessian CUDA tests.
#pragma once

#include "common/price_gradients/mixed_sensitivity_outputs.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace price_gradient_test {

struct MixedResults {
    std::vector<float> prices;
    std::vector<float> price_errors;
    std::vector<float> gradients;
    std::vector<float> gradient_errors;
    std::vector<float> diagonal_hessians;
    std::vector<float> diagonal_hessian_errors;
    std::vector<float> mixed_hessians;
    std::vector<float> mixed_hessian_errors;
};

template<typename Plan, typename WorkspaceSizer, typename Launcher>
MixedResults execute_mixed_node_graph(
    const Plan& host,
    pg::LaunchConfiguration launch,
    WorkspaceSizer workspace_size,
    Launcher launcher,
    const char* label
) {
    const auto rows = host.result_count;
    const auto sensitivity_count = host.sensitivity_count();
    const auto first_count = host.sensitivity_graph.first.size();
    const auto diagonal_count =
        host.sensitivity_graph.diagonal_second.size();
    const auto selected_capacity = rows * std::max(
        first_count, diagonal_count
    );
    const auto mixed_count = host.sensitivity_graph.mixed_second.size();
    launch.result_offset = 0U;
    launch.result_count = rows;
    launch.sensitivity_batch_size = 1U;
    launch.block_count = rows * sensitivity_count;
    const auto workspace_bytes = workspace_size(host, launch);

    DeviceInputStorage<Plan> input_storage(host);
    DeviceArray<pg::SensitivityStencil<4U>> stencils(
        rows * sensitivity_count
    );
    DeviceArray<pg::MixedSensitivityStencil> mixed_stencils(
        rows * mixed_count
    );
    DeviceArray<pg::device_preparation::Error> preparation_error(1U);
    DeviceArray<std::uint8_t> workspace(workspace_bytes);
    DeviceArray<float> prices(rows), price_errors(rows);
    DeviceArray<float> gradients(selected_capacity);
    DeviceArray<float> gradient_errors(selected_capacity);
    DeviceArray<float> diagonal_hessians(selected_capacity);
    DeviceArray<float> diagonal_hessian_errors(selected_capacity);
    DeviceArray<float> mixed_hessians(rows * mixed_count);
    DeviceArray<float> mixed_hessian_errors(rows * mixed_count);

    const typename Plan::DiagonalStencilOutputs stencil_outputs{
        stencils.data, stencils.count, preparation_error.data
    };
    const typename Plan::MixedStencilOutputs mixed_stencil_outputs{
        mixed_stencils.data, mixed_stencils.count
    };
    const pg::SensitivityOutputs outputs{
        prices.data,
        price_errors.data,
        gradients.data,
        gradient_errors.data,
        diagonal_hessians.data,
        diagonal_hessian_errors.data,
        rows,
        selected_capacity,
    };
    const pg::MixedSensitivityOutputs mixed_outputs{
        mixed_hessians.data,
        mixed_hessian_errors.data,
        rows * mixed_count,
    };

    launcher(
        host,
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
    return {
        prices.read(),
        price_errors.read(),
        gradients.read(),
        gradient_errors.read(),
        diagonal_hessians.read(),
        diagonal_hessian_errors.read(),
        mixed_hessians.read(),
        mixed_hessian_errors.read(),
    };
}

inline void require_finite_mixed_results(
    const MixedResults& results,
    const char* label
) {
    for (const auto value : results.mixed_hessians) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                std::string(label) + " Hessian is not finite."
            );
        }
    }
    for (const auto value : results.mixed_hessian_errors) {
        if (!std::isfinite(value) || value < 0.0f) {
            throw std::runtime_error(
                std::string(label) + " standard error is invalid."
            );
        }
    }
}

inline void require_diagonal_parity(
    const DiagonalResults& expected,
    const MixedResults& actual,
    const std::string& label
) {
    require_same_bits(expected.price, actual.prices, label + " price");
    require_same_bits(
        expected.price_error, actual.price_errors, label + " price error"
    );
    require_same_bits(
        expected.gradient, actual.gradients, label + " gradient"
    );
    require_same_bits(
        expected.gradient_error,
        actual.gradient_errors,
        label + " gradient error"
    );
    require_same_bits(
        expected.diagonal_hessian,
        actual.diagonal_hessians,
        label + " diagonal Hessian"
    );
    require_same_bits(
        expected.diagonal_hessian_error,
        actual.diagonal_hessian_errors,
        label + " diagonal Hessian error"
    );
}


template<
    typename Plan,
    typename MonoLauncher,
    typename WorkspaceSizer,
    typename MixedLauncher>
void require_mixed_node_graph_parity(
    const Plan& plan,
    MonoLauncher mono_launcher,
    WorkspaceSizer workspace_size,
    MixedLauncher mixed_launcher,
    std::size_t paths,
    std::uint64_t seed,
    const std::string& label
) {
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        plan.result_count,
        paths,
        128U,
        plan.result_count * plan.sensitivity_count(),
        seed,
        1U,
    };
    const auto reference = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(plan, launch, mono_launcher);
    const auto mixed = execute_mixed_node_graph(
        plan,
        launch,
        workspace_size,
        mixed_launcher,
        label.c_str()
    );
    require_diagonal_parity(reference, mixed, label);
    require_finite_mixed_results(mixed, label.c_str());
}

}  // namespace price_gradient_test
