// Shared public-API fixture for terminal diagonal-sensitivity CUDA tests.
#pragma once

#include "cuda_test_support.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"

namespace price_gradient_test {

struct DiagonalResults {
    std::vector<float> price;
    std::vector<float> price_error;
    std::vector<float> gradient;
    std::vector<float> gradient_error;
    std::vector<float> diagonal_hessian;
    std::vector<float> diagonal_hessian_error;
    std::vector<pg::SensitivityStencil<4U>> stencils;
};

template<pg::SensitivityOrders Orders, typename Plan, typename Launcher>
DiagonalResults execute_diagonal(
    const Plan& plan,
    pg::LaunchConfiguration launch,
    Launcher launcher
) {
    static_assert(pg::requests_second_v<Orders>);
    const auto rows = plan.result_count;
    const auto k = plan.sensitivity_count();
    DeviceArray<typename Plan::Model> models(plan.models);
    DeviceArray<typename Plan::Product> products(plan.products);
    DeviceArray<typename Plan::SensitivitySpec> sensitivities(plan.sensitivities);
    DeviceArray<pg::SensitivityStencil<4U>> stencils(rows*k);
    DeviceArray<equity::price_gradients::device_preparation::Error> error(1U);
    DeviceArray<float> prices(rows);
    DeviceArray<float> price_errors(rows);
    DeviceArray<float> gradients(pg::requests_first_v<Orders> ? rows*k : 0U);
    DeviceArray<float> gradient_errors(pg::requests_first_v<Orders> ? rows*k : 0U);
    DeviceArray<float> hessians(rows*k);
    DeviceArray<float> hessian_errors(rows*k);
    const typename Plan::DeviceInputs inputs{
        models.data, models.count,
        products.data, products.count,
        sensitivities.data, sensitivities.count,
    };
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
    launcher(plan, inputs, stencil_outputs, launch, outputs);
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

}  // namespace price_gradient_test
