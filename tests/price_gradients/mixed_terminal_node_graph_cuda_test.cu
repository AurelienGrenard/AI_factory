// Full-Hessian Heston pilot for the selected terminal node graph.
#include "common/monte_carlo/price_gradients/device_prepared_terminal_diagonal_kernel.cuh"
#include "common/monte_carlo/price_gradients/terminal_node_graph/mixed_launcher.cuh"
#include "model/equity/markovian/heston/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "product/european_option/price_gradients/monte_carlo_policy.cuh"
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
namespace mcpg = monte_carlo::price_gradients;
namespace heston = model::equity::heston;

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

template<typename Plan>
MixedResults run_mixed(
    const Plan& host,
    pg::LaunchConfiguration launch,
    mcpg::TerminalNodeGraphConfiguration graph_configuration
) {
    constexpr std::size_t maximum_sensitivities = 3U;
    constexpr std::size_t maximum_mixed = 3U;
    constexpr unsigned int group_size = 16U;
    constexpr unsigned int nodes_per_worker = 2U;
    using Dynamics = heston::price_gradients::CoupledDynamics;
    using ProductPolicy =
        product::EuropeanOptionGradientPathPolicy<OptionSide::call>;
    using Preparation = typename Plan::PreparationPolicy;
    using NodePolicy = mcpg::TerminalNodePolicy<
        Dynamics, ProductPolicy, Preparation
    >;
    using Workspace = mcpg::MixedTerminalNodeGraphWorkspace<NodePolicy>;

    const auto rows = host.result_count;
    const auto sensitivity_count = host.sensitivity_count();
    const auto mixed_count = host.sensitivity_graph.mixed_second.size();
    const auto requirements =
        mcpg::mixed_terminal_node_graph_workspace_requirements(
            sensitivity_count,
            host.sensitivity_graph,
            launch.threads_per_block,
            graph_configuration
        );

    DeviceArray<typename Plan::Model> models(host.models);
    DeviceArray<typename Plan::Product> products(host.products);
    DeviceArray<typename Plan::SensitivitySpec> sensitivities(
        host.sensitivities
    );
    DeviceArray<std::uint16_t> first(host.sensitivity_graph.first);
    DeviceArray<std::uint16_t> diagonal(
        host.sensitivity_graph.diagonal_second
    );
    DeviceArray<pg::SensitivityPair> mixed(
        host.sensitivity_graph.mixed_second
    );
    DeviceArray<pg::SensitivityCoordinateUse> coordinate_uses(
        host.sensitivity_graph.coordinate_uses
    );
    DeviceArray<pg::SensitivityStencil<4U>> stencils(
        rows * sensitivity_count
    );
    DeviceArray<pg::MixedSensitivityStencil> mixed_stencils(
        rows * mixed_count
    );
    DeviceArray<pg::device_preparation::Error> preparation_error(1U);
    DeviceArray<typename NodePolicy::NodeValue> node_values(
        requirements.node_values
    );
    DeviceArray<typename NodePolicy::Metadata> node_metadata(
        requirements.node_metadata
    );
    DeviceArray<mcpg::SensitivityNodeIndices<4U>> axis_indices(
        requirements.axis_node_indices
    );
    DeviceArray<mcpg::MixedSensitivityNodeIndices> mixed_indices(
        requirements.mixed_node_indices
    );
    DeviceArray<std::uint8_t> row_status(requirements.row_status);
    DeviceArray<reductions::MomentSums> thread_moments(
        requirements.thread_moments
    );
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
    const pg::DeviceSensitivityGraph device_graph{
        first.data,
        first.count,
        first.count,
        diagonal.data,
        diagonal.count,
        diagonal.count,
        mixed.data,
        mixed.count,
        mixed.count,
        coordinate_uses.data,
        coordinate_uses.count,
        coordinate_uses.count,
        host.sensitivity_graph.node_capacity,
    };
    const Workspace workspace{
        node_values.data,
        node_values.count,
        node_metadata.data,
        node_metadata.count,
        axis_indices.data,
        axis_indices.count,
        mixed_indices.data,
        mixed_indices.count,
        row_status.data,
        row_status.count,
        thread_moments.data,
        thread_moments.count,
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
    const mcpg::DevicePreparedMixedStencilOutputs mixed_stencil_outputs{
        mixed_stencils.data, mixed_stencils.count
    };

    mcpg::launch_device_prepared_terminal_mixed_node_graph<
        Dynamics,
        ProductPolicy,
        Preparation,
        maximum_sensitivities,
        maximum_mixed,
        group_size,
        nodes_per_worker
    >(
        inputs,
        mcpg::make_device_prepared_plan(host),
        host.sensitivity_graph,
        device_graph,
        launch,
        graph_configuration,
        workspace,
        outputs,
        mixed_outputs,
        stencil_outputs,
        mixed_stencil_outputs,
        "test.heston_mixed_terminal_node_graph",
        "full_hessian"
    );
    check_cuda(cudaDeviceSynchronize(), "Heston mixed terminal graph");
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
    const auto mixed = run_mixed(plan, launch, {2U, paths, 1U});
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
}

}  // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        check_heston();
        std::cout
            << "Heston full Hessian preserves price/gradient/diagonal bits\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
