// Full-Hessian Heston pilot for the selected terminal node graph.
#include "common/monte_carlo/price_gradients/device_prepared_terminal_diagonal_kernel.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ai_factory::workbench;
using price_gradient_test::DeviceArray;
using price_gradient_test::require;
namespace pg = price_gradients;
namespace heston = model::equity::heston;
namespace merton = model::equity::merton;

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

void same_bits(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const char* label
) {
    require(expected.size() == actual.size(), "Mixed parity size mismatch.");
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(expected[index])
            == std::bit_cast<std::uint32_t>(actual[index])) {
            continue;
        }
        throw std::runtime_error(
            std::string(label) + " differs at " + std::to_string(index)
        );
    }
}

template<typename Plan, typename WorkspaceSizer, typename Launcher>
MixedResults run_mixed(
    const Plan& host,
    pg::LaunchConfiguration launch,
    WorkspaceSizer workspace_size,
    Launcher launcher,
    const char* label
) {
    const auto rows = host.result_count;
    const auto sensitivity_count = host.sensitivity_count();
    const auto mixed_count = host.sensitivity_graph.mixed_second.size();
    const auto workspace_bytes = workspace_size(host, launch);

    DeviceArray<typename Plan::Model> models(host.models);
    DeviceArray<typename Plan::Product> products(host.products);
    DeviceArray<typename Plan::SensitivitySpec> sensitivities(
        host.sensitivities
    );
    DeviceArray<pg::SensitivityStencil<4U>> stencils(
        rows * sensitivity_count
    );
    DeviceArray<pg::MixedSensitivityStencil> mixed_stencils(
        rows * mixed_count
    );
    DeviceArray<pg::device_preparation::Error> preparation_error(1U);
    DeviceArray<std::uint8_t> workspace(workspace_bytes);
    DeviceArray<float> prices(rows), price_errors(rows);
    DeviceArray<float> gradients(rows * host.sensitivity_graph.first.size());
    DeviceArray<float> gradient_errors(
        rows * host.sensitivity_graph.first.size()
    );
    DeviceArray<float> diagonal_hessians(
        rows * host.sensitivity_graph.diagonal_second.size()
    );
    DeviceArray<float> diagonal_hessian_errors(
        rows * host.sensitivity_graph.diagonal_second.size()
    );
    DeviceArray<float> mixed_hessians(rows * mixed_count);
    DeviceArray<float> mixed_hessian_errors(rows * mixed_count);

    const typename Plan::DeviceInputs inputs{
        models.data,
        models.count,
        products.data,
        products.count,
        sensitivities.data,
        sensitivities.count,
    };
    const pg::SensitivityOutputs outputs{
        prices.data,
        price_errors.data,
        gradients.data,
        gradient_errors.data,
        diagonal_hessians.data,
        diagonal_hessian_errors.data,
        rows,
        rows * sensitivity_count,
    };
    const pg::MixedSensitivityOutputs mixed_outputs{
        mixed_hessians.data,
        mixed_hessian_errors.data,
        rows * mixed_count,
    };
    const typename Plan::DiagonalStencilOutputs stencil_outputs{
        stencils.data, stencils.count, preparation_error.data
    };
    const pg::MixedSensitivityStencilOutputs mixed_stencil_outputs{
        mixed_stencils.data, mixed_stencils.count
    };

    launcher(
        host,
        inputs,
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
            "Mixed preparation failed at row "
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

void check_heston() {
    const std::vector<heston::ModelParameters> models{
        {1.0f, 0.01f, 0.0f, 0.04f, 1.5f, 0.04f, 0.3f, -0.7f},
        {1.1f, -0.01f, 0.02f, 0.06f, 0.8f, 0.05f, 0.4f, -0.3f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 16U}, {1.05f, 24U},
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.initial_variance", {
            0.001f, pg::BumpScale::absolute
        }},
        {"model.rho", {0.002f, pg::BumpScale::absolute}},
        {"product.strike", {0.005f}},
    }};
    const auto plan = heston::prepare_heston_european_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {},
        selection,
        pg::SensitivityRequest::full_hessian()
    );
    require(plan.sensitivity_graph.node_capacity == 22U
            && plan.sensitivity_graph.output_count() == 10U,
            "Heston full-Hessian host graph is incorrect.");

    constexpr std::size_t paths = 4096U;
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        models.size(),
        paths,
        128U,
        models.size() * selection.sensitivities.size(),
        9127U,
        1U,
    };
    const auto reference = price_gradient_test::execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        launch,
        heston::launch_heston_european_option_diagonal_sensitivities_cuda<
            OptionSide::call, pg::SensitivityOrders::first_and_second
        >
    );
    const auto mixed = run_mixed(
        plan,
        launch,
        heston::heston_european_option_mixed_node_graph_workspace_bytes<
            OptionSide::call
        >,
        heston::launch_heston_european_option_mixed_node_graph_sensitivities_cuda<
            OptionSide::call
        >,
        "Heston mixed terminal graph"
    );
    same_bits(reference.price, mixed.prices, "central price");
    same_bits(reference.price_error, mixed.price_errors, "central error");
    same_bits(reference.gradient, mixed.gradients, "gradient");
    same_bits(reference.gradient_error, mixed.gradient_errors,
              "gradient error");
    same_bits(reference.diagonal_hessian, mixed.diagonal_hessians,
              "diagonal Hessian");
    same_bits(reference.diagonal_hessian_error,
              mixed.diagonal_hessian_errors,
              "diagonal Hessian error");
    for (const auto value : mixed.mixed_hessians) {
        require(std::isfinite(value), "Mixed Heston Hessian is not finite.");
    }
    for (const auto value : mixed.mixed_hessian_errors) {
        require(std::isfinite(value) && value >= 0.0f,
                "Mixed Heston Hessian error is invalid.");
    }

    const auto sparse_plan =
        heston::prepare_heston_european_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            {},
            selection,
            pg::SensitivityRequest::selected(
                {2U}, {0U}, {{2U, 1U}}
            )
        );
    require(sparse_plan.sensitivity_graph.node_capacity == 12U,
            "Sparse Heston graph did not remove unused nodes.");
    const auto sparse = run_mixed(
        sparse_plan,
        launch,
        heston::heston_european_option_mixed_node_graph_workspace_bytes<
            OptionSide::call
        >,
        heston::launch_heston_european_option_mixed_node_graph_sensitivities_cuda<
            OptionSide::call
        >,
        "Heston sparse mixed terminal graph"
    );
    same_bits(mixed.prices, sparse.prices, "sparse central price");
    same_bits(mixed.price_errors, sparse.price_errors,
              "sparse central error");
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same_bits(
            {mixed.gradients[row * 3U + 2U]},
            {sparse.gradients[row]},
            "sparse selected gradient"
        );
        same_bits(
            {mixed.diagonal_hessians[row * 3U]},
            {sparse.diagonal_hessians[row]},
            "sparse selected diagonal Hessian"
        );
        same_bits(
            {mixed.mixed_hessians[row * 3U + 2U]},
            {sparse.mixed_hessians[row]},
            "sparse selected mixed Hessian"
        );
    }
}

void check_merton() {
    const std::vector<merton::ModelParameters> models{
        {1.05f, 0.03f, 0.01f, 0.20f, 0.50f, -0.10f, 0.25f},
        {1.00f, -0.01f, 0.00f, 0.35f, 2.00f, 0.02f, 0.40f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.00f, 32U}, {1.10f, 48U},
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.jump_intensity", {
            0.05f, pg::BumpScale::absolute
        }},
        {"model.jump_log_mean", {
            0.002f, pg::BumpScale::absolute
        }},
        {"product.strike", {0.005f}},
    }};
    const auto plan = merton::prepare_merton_european_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {},
        selection,
        pg::SensitivityRequest::full_hessian()
    );
    constexpr std::size_t paths = 4096U;
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        models.size(),
        paths,
        128U,
        models.size() * selection.sensitivities.size(),
        8273U,
        1U,
    };
    const auto reference = price_gradient_test::execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        launch,
        merton::launch_merton_european_option_diagonal_sensitivities_cuda<
            OptionSide::call, pg::SensitivityOrders::first_and_second
        >
    );
    const auto mixed = run_mixed(
        plan,
        launch,
        merton::merton_european_option_mixed_node_graph_workspace_bytes<
            OptionSide::call
        >,
        merton::launch_merton_european_option_mixed_node_graph_sensitivities_cuda<
            OptionSide::call
        >,
        "Merton mixed terminal graph"
    );
    same_bits(reference.price, mixed.prices, "Merton central price");
    same_bits(reference.price_error, mixed.price_errors,
              "Merton central error");
    same_bits(reference.gradient, mixed.gradients, "Merton gradient");
    same_bits(reference.gradient_error, mixed.gradient_errors,
              "Merton gradient error");
    same_bits(reference.diagonal_hessian, mixed.diagonal_hessians,
              "Merton diagonal Hessian");
    same_bits(reference.diagonal_hessian_error,
              mixed.diagonal_hessian_errors,
              "Merton diagonal Hessian error");
    for (const auto value : mixed.mixed_hessians) {
        require(std::isfinite(value), "Mixed Merton Hessian is not finite.");
    }
    for (const auto value : mixed.mixed_hessian_errors) {
        require(std::isfinite(value) && value >= 0.0f,
                "Mixed Merton Hessian error is invalid.");
    }
}

}  // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        check_heston();
        check_merton();
        std::cout
            << "Heston and Merton full Hessians preserve "
               "price/gradient/diagonal bits\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
