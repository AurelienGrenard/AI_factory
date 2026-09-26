// Full-Hessian Heston pilot for the selected terminal node graph.
#include "common/monte_carlo/price_gradients/device_prepared_terminal_diagonal_kernel.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "mixed_node_graph_cuda_test_support.cuh"

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

using price_gradient_test::execute_mixed_node_graph;
using price_gradient_test::require_diagonal_parity;
using price_gradient_test::require_finite_mixed_results;
using price_gradient_test::require_same_bits;

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
    const auto mixed = execute_mixed_node_graph(
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
    require_same_bits(reference.price, mixed.prices, "central price");
    require_same_bits(reference.price_error, mixed.price_errors, "central error");
    require_same_bits(reference.gradient, mixed.gradients, "gradient");
    require_same_bits(reference.gradient_error, mixed.gradient_errors,
              "gradient error");
    require_same_bits(reference.diagonal_hessian, mixed.diagonal_hessians,
              "diagonal Hessian");
    require_same_bits(reference.diagonal_hessian_error,
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
    const auto sparse = execute_mixed_node_graph(
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
    require_same_bits(mixed.prices, sparse.prices, "sparse central price");
    require_same_bits(mixed.price_errors, sparse.price_errors,
              "sparse central error");
    for (std::size_t row = 0U; row < models.size(); ++row) {
        require_same_bits(
            {mixed.gradients[row * 3U + 2U]},
            {sparse.gradients[row]},
            "sparse selected gradient"
        );
        require_same_bits(
            {mixed.diagonal_hessians[row * 3U]},
            {sparse.diagonal_hessians[row]},
            "sparse selected diagonal Hessian"
        );
        require_same_bits(
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
    const auto mixed = execute_mixed_node_graph(
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
    require_same_bits(reference.price, mixed.prices, "Merton central price");
    require_same_bits(reference.price_error, mixed.price_errors,
              "Merton central error");
    require_same_bits(reference.gradient, mixed.gradients, "Merton gradient");
    require_same_bits(reference.gradient_error, mixed.gradient_errors,
              "Merton gradient error");
    require_same_bits(reference.diagonal_hessian, mixed.diagonal_hessians,
              "Merton diagonal Hessian");
    require_same_bits(reference.diagonal_hessian_error,
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
