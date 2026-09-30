// Shared public-API fixture for terminal diagonal-sensitivity CUDA tests.
#pragma once

#include "cuda_test_support.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"

#include <bit>
#include <cstdint>
#include <string>

namespace price_gradient_test {

template<typename Plan, bool HasCurve = requires { typename Plan::Curve; }>
struct DeviceInputStorage;

template<typename Plan>
struct DeviceInputStorage<Plan, false> {
    DeviceArray<typename Plan::Model> models;
    DeviceArray<typename Plan::Product> products;
    DeviceArray<typename Plan::SensitivitySpec> sensitivities;

    explicit DeviceInputStorage(const Plan& plan)
        : models(plan.models),
          products(plan.products),
          sensitivities(plan.sensitivities) {}

    typename Plan::DeviceInputs inputs() const {
        return {
            models.data,
            models.count,
            products.data,
            products.count,
            sensitivities.data,
            sensitivities.count,
        };
    }
};

template<typename Plan>
struct DeviceInputStorage<Plan, true> {
    DeviceArray<typename Plan::Model> models;
    DeviceArray<typename Plan::Curve> curves;
    DeviceArray<typename Plan::Product> products;
    DeviceArray<typename Plan::SensitivitySpec> sensitivities;

    explicit DeviceInputStorage(const Plan& plan)
        : models(plan.models),
          curves(plan.curves),
          products(plan.products),
          sensitivities(plan.sensitivities) {}

    typename Plan::DeviceInputs inputs() const {
        return {
            models.data,
            models.count,
            curves.data,
            curves.count,
            products.data,
            products.count,
            sensitivities.data,
            sensitivities.count,
        };
    }
};

struct DiagonalResults {
    std::vector<float> price;
    std::vector<float> price_error;
    std::vector<float> gradient;
    std::vector<float> gradient_error;
    std::vector<float> diagonal_hessian;
    std::vector<float> diagonal_hessian_error;
    std::vector<pg::SensitivityStencil<4U>> stencils;
};

inline void require_same_bits(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const std::string& label
) {
    require(expected.size() == actual.size(), "Result size mismatch.");
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(expected[index])
            == std::bit_cast<std::uint32_t>(actual[index])) {
            continue;
        }
        throw std::runtime_error(
            label + " differs at output " + std::to_string(index)
            + ": expected=" + std::to_string(expected[index])
            + " (bits=" + std::to_string(
                std::bit_cast<std::uint32_t>(expected[index])
            ) + "), actual=" + std::to_string(actual[index])
            + " (bits=" + std::to_string(
                std::bit_cast<std::uint32_t>(actual[index])
            ) + ")"
        );
    }
}

inline void require_same_stencils(
    const std::vector<pg::SensitivityStencil<4U>>& expected,
    const std::vector<pg::SensitivityStencil<4U>>& actual,
    const std::string& label
) {
    require(expected.size() == actual.size(), "Stencil size mismatch.");
    const auto same_float = [](float first, float second) {
        return std::bit_cast<std::uint32_t>(first)
            == std::bit_cast<std::uint32_t>(second);
    };
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        const auto& first = expected[index];
        const auto& second = actual[index];
        if (first.kind != second.kind
            || first.node_count != second.node_count
            || !same_float(first.displacement, second.displacement)
            || !same_float(first.represented_width, second.represented_width)) {
            throw std::runtime_error(label + " stencil header differs.");
        }
        for (std::size_t node = 0U; node < 4U; ++node) {
            if (!same_float(
                    first.parameter_values[node],
                    second.parameter_values[node]
                )
                || !same_float(
                    first.second_weights[node],
                    second.second_weights[node]
                )) {
                throw std::runtime_error(
                    label + " stencil node differs at stencil "
                    + std::to_string(index) + ", node "
                    + std::to_string(node)
                );
            }
        }
        for (std::size_t endpoint = 0U; endpoint < 2U; ++endpoint) {
            if (!same_float(
                    first.first_endpoint_weights[endpoint],
                    second.first_endpoint_weights[endpoint]
                )) {
                throw std::runtime_error(
                    label + " stencil endpoint differs at stencil "
                    + std::to_string(index) + ", endpoint "
                    + std::to_string(endpoint)
                );
            }
        }
    }
}

inline void require_same_diagonal_results(
    const DiagonalResults& expected,
    const DiagonalResults& actual,
    const std::string& label
) {
    require_same_stencils(expected.stencils, actual.stencils, label);
    require_same_bits(expected.price, actual.price, label + " price");
    require_same_bits(
        expected.price_error, actual.price_error, label + " price error"
    );
    require_same_bits(expected.gradient, actual.gradient, label + " gradient");
    require_same_bits(
        expected.gradient_error,
        actual.gradient_error,
        label + " gradient error"
    );
    require_same_bits(
        expected.diagonal_hessian,
        actual.diagonal_hessian,
        label + " diagonal Hessian"
    );
    require_same_bits(
        expected.diagonal_hessian_error,
        actual.diagonal_hessian_error,
        label + " diagonal Hessian error"
    );
}

template<
    pg::SensitivityOrders Orders,
    typename Plan,
    typename WorkspaceBytes,
    typename Invocation>
DiagonalResults execute_diagonal_with_workspace(
    const Plan& plan,
    pg::LaunchConfiguration launch,
    WorkspaceBytes workspace_bytes,
    Invocation invoke
) {
    static_assert(pg::requests_second_v<Orders>);
    const auto rows = plan.result_count;
    const auto k = plan.sensitivity_count();
    DeviceInputStorage<Plan> input_storage(plan);
    DeviceArray<pg::SensitivityStencil<4U>> stencils(rows*k);
    DeviceArray<equity::price_gradients::device_preparation::Error> error(1U);
    DeviceArray<float> prices(rows);
    DeviceArray<float> price_errors(rows);
    DeviceArray<float> gradients(pg::requests_first_v<Orders> ? rows*k : 0U);
    DeviceArray<float> gradient_errors(pg::requests_first_v<Orders> ? rows*k : 0U);
    DeviceArray<float> hessians(rows*k);
    DeviceArray<float> hessian_errors(rows*k);
    const auto inputs = input_storage.inputs();
    const typename Plan::DiagonalStencilOutputs stencil_outputs{
        stencils.data, stencils.count, error.data
    };
    const pg::SensitivityOutputs outputs{
        prices.data, price_errors.data,
        gradients.data, gradient_errors.data,
        hessians.data, hessian_errors.data,
        rows, rows*k,
    };
    launch.result_offset = 0U;
    launch.result_count = rows;
    if (launch.method == pg::PricingMethod::monte_carlo) {
        launch.sensitivity_batch_size = 1U;
        launch.block_count = rows*(k == 0U ? 1U : k);
    }
    DeviceArray<std::uint8_t> workspace(workspace_bytes(plan, launch));
    invoke(
        plan,
        inputs,
        stencil_outputs,
        launch,
        outputs,
        workspace.data,
        workspace.count
    );
    check_cuda(cudaDeviceSynchronize(), "Terminal diagonal-sensitivity test");
    const auto preparation_error = error.read()[0U];
    if (preparation_error.code != 0) {
        throw std::runtime_error(
            "Terminal diagonal preparation failed at row "
            + std::to_string(preparation_error.row)
            + ", sensitivity "
            + std::to_string(preparation_error.sensitivity)
            + ", code "
            + std::to_string(preparation_error.code)
        );
    }
    return {
        prices.read(), price_errors.read(),
        gradients.read(), gradient_errors.read(),
        hessians.read(), hessian_errors.read(), stencils.read(),
    };
}

template<pg::SensitivityOrders Orders, typename Plan, typename Launcher>
DiagonalResults execute_diagonal(
    const Plan& plan,
    pg::LaunchConfiguration launch,
    Launcher launcher
) {
    return execute_diagonal_with_workspace<Orders>(
        plan,
        launch,
        [](const auto&, const auto&) { return std::size_t{0U}; },
        [launcher](
            const auto& current_plan,
            auto inputs,
            auto stencils,
            const auto& configuration,
            auto outputs,
            void*,
            std::size_t
        ) {
            launcher(
                current_plan,
                inputs,
                stencils,
                configuration,
                outputs
            );
        }
    );
}

template<
    pg::SensitivityOrders Orders,
    typename Plan,
    typename WorkspaceBytes,
    typename Launcher>
DiagonalResults execute_diagonal_node_graph(
    const Plan& plan,
    pg::LaunchConfiguration launch,
    WorkspaceBytes workspace_bytes,
    Launcher launcher
) {
    return execute_diagonal_with_workspace<Orders>(
        plan,
        launch,
        workspace_bytes,
        [launcher](
            const auto& current_plan,
            auto inputs,
            auto stencils,
            const auto& configuration,
            auto outputs,
            void* workspace,
            std::size_t bytes
        ) {
            launcher(
                current_plan,
                inputs,
                stencils,
                configuration,
                outputs,
                workspace,
                bytes
            );
        }
    );
}


template<typename Plan, typename MonoLaunch>
void require_price_only_parity(
    const Plan& sensitivity_plan,
    const Plan& price_only_plan,
    MonoLaunch mono_launch,
    std::size_t paths,
    std::uint64_t seed,
    const std::string& label
) {
    pg::LaunchConfiguration configuration{
        pg::PricingMethod::monte_carlo,
        0U,
        sensitivity_plan.result_count,
        paths,
        128U,
        sensitivity_plan.result_count
            * sensitivity_plan.sensitivity_count(),
        seed,
        1U,
    };
    const auto sensitivities = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(sensitivity_plan, configuration, mono_launch);
    const auto price_only = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(price_only_plan, configuration, mono_launch);
    require_same_bits(
        price_only.price, sensitivities.price, label + " price"
    );
    require_same_bits(
        price_only.price_error,
        sensitivities.price_error,
        label + " price error"
    );
}

template<
    typename Plan,
    typename MonoLaunch,
    typename GraphWorkspaceBytes,
    typename GraphLaunch>
void require_mono_node_graph_parity(
    const Plan& plan,
    MonoLaunch mono_launch,
    GraphWorkspaceBytes graph_workspace_bytes,
    GraphLaunch graph_launch,
    std::size_t paths,
    std::uint64_t seed,
    const std::string& label
) {
    pg::LaunchConfiguration configuration{
        pg::PricingMethod::monte_carlo,
        0U,
        plan.result_count,
        paths,
        128U,
        plan.result_count * plan.sensitivity_count(),
        seed,
        1U,
    };
    const auto mono = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(plan, configuration, mono_launch);
    const auto graph = execute_diagonal_node_graph<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        configuration,
        graph_workspace_bytes,
        graph_launch
    );
    require_same_diagonal_results(mono, graph, label);
}

}  // namespace price_gradient_test
