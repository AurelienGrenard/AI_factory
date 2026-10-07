// Bitwise parity of the terminal three-kernel graph against the mono engine.
#include "common/monte_carlo/price_gradients/device_prepared_terminal_diagonal_kernel.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_terminal_node_graph.cuh"
#include "model/equity/markovian/bates/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/bates/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/cev/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/cev/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "product/european_option/price_gradients/monte_carlo_policy.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ai_factory::workbench;
using price_gradient_test::DeviceArray;
using price_gradient_test::DiagonalResults;
using price_gradient_test::require;
namespace pg = price_gradients;
namespace mcpg = monte_carlo::price_gradients;
namespace heston = model::equity::heston;
namespace bates = model::equity::bates;
namespace cev = model::equity::cev;
namespace merton = model::equity::merton;

struct UnboundedMonoTuning {
    static constexpr bool kLaunchBoundsEnabled = false;
    static constexpr unsigned int kThreadsPerBlock = 128U;
    static constexpr unsigned int kMinimumBlocksPerMultiprocessor = 6U;
};

void same_bits(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const char* label
) {
    require(expected.size() == actual.size(), "Result size mismatch.");
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(expected[index])
            == std::bit_cast<std::uint32_t>(actual[index])) {
            continue;
        }
        throw std::runtime_error(
            std::string(label) + " differs at "
            + std::to_string(index)
        );
    }
}

void same_stencils(
    const std::vector<pg::SensitivityStencil<4U>>& expected,
    const std::vector<pg::SensitivityStencil<4U>>& actual
) {
    require(expected.size() == actual.size(), "Stencil size mismatch.");
    const auto same_float = [](float first, float second) {
        return std::bit_cast<std::uint32_t>(first)
            == std::bit_cast<std::uint32_t>(second);
    };
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        const auto& first = expected[index];
        const auto& second = actual[index];
        require(first.kind == second.kind, "Stencil kind mismatch.");
        require(first.node_count == second.node_count,
                "Stencil node-count mismatch.");
        require(same_float(first.displacement, second.displacement),
                "Stencil displacement mismatch.");
        require(same_float(first.represented_width, second.represented_width),
                "Stencil width mismatch.");
        for (std::size_t node = 0U; node < 4U; ++node) {
            require(
                same_float(
                    first.parameter_values[node],
                    second.parameter_values[node]
                ),
                "Stencil node mismatch."
            );
            require(
                same_float(
                    first.second_weights[node],
                    second.second_weights[node]
                ),
                "Stencil second weight mismatch."
            );
        }
        for (std::size_t endpoint = 0U; endpoint < 2U; ++endpoint) {
            require(
                same_float(
                    first.first_endpoint_weights[endpoint],
                    second.first_endpoint_weights[endpoint]
                ),
                "Stencil first weight mismatch."
            );
        }
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Plan,
    std::size_t MaximumSensitivities = 7U,
    unsigned int GroupSize = 16U,
    unsigned int NodesPerWorker = 2U>
DiagonalResults run_graph(
    const Plan& host,
    pg::LaunchConfiguration launch,
    mcpg::TerminalNodeGraphConfiguration graph
) {
    static_assert(pg::requests_second_v<Orders>);
    using Preparation = typename Plan::PreparationPolicy;
    using NodePolicy =
        mcpg::TerminalNodePolicy<Dynamics, ProductPolicy, Preparation>;
    using Workspace = mcpg::TerminalNodeGraphWorkspace<NodePolicy>;

    const auto rows = host.result_count;
    const auto sensitivity_count = host.sensitivity_count();
    const auto requirements =
        mcpg::terminal_node_graph_workspace_requirements<
            Orders, MaximumSensitivities
        >(
            sensitivity_count,
            launch.threads_per_block,
            graph
        );

    DeviceArray<typename Plan::Model> models(host.models);
    DeviceArray<typename Plan::Product> products(host.products);
    DeviceArray<typename Plan::SensitivitySpec> sensitivities(
        host.sensitivities
    );
    DeviceArray<pg::SensitivityStencil<4U>> stencils(
        rows * sensitivity_count
    );
    DeviceArray<pg::device_preparation::Error> error(1U);
    DeviceArray<typename NodePolicy::NodeValue> node_values(
        requirements.node_values
    );
    DeviceArray<typename NodePolicy::Metadata> node_metadata(
        requirements.node_metadata
    );
    DeviceArray<mcpg::SensitivityNodeIndices<4U>> node_indices(
        requirements.node_indices
    );
    DeviceArray<std::uint8_t> row_status(requirements.row_status);
    DeviceArray<reductions::MomentSums> thread_moments(
        requirements.thread_moments
    );
    DeviceArray<float> prices(rows), price_errors(rows);
    DeviceArray<float> gradients(
        pg::requests_first_v<Orders> ? rows * sensitivity_count : 0U
    );
    DeviceArray<float> gradient_errors(
        pg::requests_first_v<Orders> ? rows * sensitivity_count : 0U
    );
    DeviceArray<float> hessians(rows * sensitivity_count);
    DeviceArray<float> hessian_errors(rows * sensitivity_count);

    const typename Plan::DeviceInputs inputs{
        models.data,
        models.count,
        products.data,
        products.count,
        sensitivities.data,
        sensitivities.count,
    };
    const typename Plan::DiagonalStencilOutputs stencil_outputs{
        stencils.data, stencils.count, error.data
    };
    const Workspace workspace{
        node_values.data,
        node_values.count,
        node_metadata.data,
        node_metadata.count,
        node_indices.data,
        node_indices.count,
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
        hessians.data,
        hessian_errors.data,
        rows,
        rows * sensitivity_count,
    };
    const mcpg::DevicePreparedPlan plan{
        host.time,
        host.construction,
        host.models.size(),
        host.products.size(),
        host.result_count,
        host.sensitivity_count(),
    };

    mcpg::launch_device_prepared_terminal_node_graph<
        Orders,
        Dynamics,
        ProductPolicy,
        Preparation,
        MaximumSensitivities,
        GroupSize,
        NodesPerWorker
    >(
        inputs,
        plan,
        launch,
        graph,
        workspace,
        outputs,
        stencil_outputs,
        "test.terminal_sensitivity_node_graph",
        "device_prepared"
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "Terminal sensitivity node-graph test"
    );
    const auto preparation_error = error.read()[0U];
    if (preparation_error.code != 0) {
        throw std::runtime_error(
            "Node-graph preparation failed at row "
            + std::to_string(preparation_error.row)
            + ", sensitivity "
            + std::to_string(preparation_error.sensitivity)
            + ", code "
            + std::to_string(preparation_error.code)
        );
    }
    return {
        prices.read(),
        price_errors.read(),
        gradients.read(),
        gradient_errors.read(),
        hessians.read(),
        hessian_errors.read(),
        stencils.read(),
    };
}

void compare(
    const DiagonalResults& reference,
    const DiagonalResults& graph
) {
    same_stencils(reference.stencils, graph.stencils);
    same_bits(reference.price, graph.price, "price");
    same_bits(reference.price_error, graph.price_error, "price error");
    same_bits(reference.gradient, graph.gradient, "gradient");
    same_bits(
        reference.gradient_error,
        graph.gradient_error,
        "gradient error"
    );
    same_bits(
        reference.diagonal_hessian,
        graph.diagonal_hessian,
        "diagonal Hessian"
    );
    same_bits(
        reference.diagonal_hessian_error,
        graph.diagonal_hessian_error,
        "diagonal Hessian error"
    );
}

template<OptionSide Side>
void check_heston(
    std::size_t paths,
    PriceConstruction construction
) {
    const std::vector<heston::ModelParameters> models{
        {0.75f, 0.01f, 0.00f, 0.03f, 1.50f, 0.04f, 0.30f, -0.70f},
        {1.20f, 0.03f, 0.01f, 0.06f, 0.80f, 0.05f, 0.40f, -0.30f},
        {1.40f, 0.04f, 0.02f, 0.04f, 1.00f, 0.04f, 0.25f,  0.20f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {0.80f, 16U}, {1.20f, 12U}, {1.10f, 8U},
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f}},
        {"model.initial_variance", {
            .001f, pg::BumpScale::absolute
        }},
        {"model.kappa", {.005f}},
        {"model.theta", {.005f}},
        {"model.gamma", {.005f}},
        {"model.rho", {.002f, pg::BumpScale::absolute}},
        {"product.strike", {.002f}},
    }};
    const auto plan = heston::prepare_heston_european_option_sensitivities(
        models,
        products,
        construction,
        {},
        selection,
        {pg::SensitivityOrders::first_and_second}
    );
    const auto rows = construction == PriceConstruction::Aligned
        ? models.size()
        : models.size() * products.size();
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        paths,
        128U,
        rows * selection.sensitivities.size(),
        1709U,
        1U,
    };
    const auto reference = price_gradient_test::execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        launch,
        heston::launch_heston_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto unbounded =
        price_gradient_test::execute_diagonal<
            pg::SensitivityOrders::first_and_second
        >(
            plan,
            launch,
            [](
                const heston::EuropeanOptionPriceGradientPlan& current,
                heston::EuropeanOptionPriceGradientPlan::DeviceInputs inputs,
                heston::EuropeanOptionPriceGradientPlan::
                    DiagonalStencilOutputs stencil_outputs,
                const pg::LaunchConfiguration& configuration,
                pg::SensitivityOutputs outputs
            ) {
                const mcpg::DevicePreparedPlan device_plan{
                    current.time,
                    current.construction,
                    current.models.size(),
                    current.products.size(),
                    current.result_count,
                    current.sensitivity_count(),
                };
                mcpg::
                    launch_device_prepared_terminal_diagonal_sensitivities<
                        pg::SensitivityOrders::first_and_second,
                        heston::price_gradients::CoupledDynamics,
                        product::EuropeanOptionGradientPathPolicy<Side>,
                        typename heston::EuropeanOptionPriceGradientPlan::
                            Preparation,
                        UnboundedMonoTuning
                    >(
                        inputs,
                        device_plan,
                        configuration,
                        outputs,
                        stencil_outputs,
                        "test.heston.unbounded_mono",
                        "gradient_and_diagonal_hessian"
                    );
            }
        );
    compare(reference, unbounded);

    const auto graph = run_graph<
        pg::SensitivityOrders::first_and_second,
        heston::price_gradients::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>
    >(
        plan,
        launch,
        {2U, 512U, 3U}
    );
    compare(reference, graph);

    const auto second_plan =
        heston::prepare_heston_european_option_sensitivities(
            models,
            products,
            construction,
            {},
            selection,
            {pg::SensitivityOrders::second}
        );
    const auto second_reference = price_gradient_test::execute_diagonal<
        pg::SensitivityOrders::second
    >(
        second_plan,
        launch,
        heston::launch_heston_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::second
        >
    );
    const auto second_graph = run_graph<
        pg::SensitivityOrders::second,
        heston::price_gradients::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>
    >(
        second_plan,
        launch,
        {2U, 512U, 3U}
    );
    compare(second_reference, second_graph);
}

template<OptionSide Side>
void check_cev(std::size_t paths) {
    const std::vector<cev::ModelParameters> models{
        {1.05f, .03f, .01f, .2f, .7f},
        {1.f, 0.f, 0.f, .3f, .999f},
        {.01f, 0.f, 0.f, 2.f, .5f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.f, 16U}, {1.f, 8U}, {.02f, 4U},
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f}},
        {"model.risk_free_rate", {
            .0001f, pg::BumpScale::absolute
        }},
        {"model.dividend_yield", {
            .0001f, pg::BumpScale::absolute
        }},
        {"model.sigma", {.005f}},
        {"model.beta", {.002f, pg::BumpScale::absolute}},
        {"product.strike", {.005f}},
        {"product.maturity_years", {
            1.f / 504.f, pg::BumpScale::absolute
        }},
    }};
    const auto plan = cev::prepare_cev_european_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {},
        selection,
        {pg::SensitivityOrders::first_and_second}
    );
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        models.size(),
        paths,
        128U,
        models.size() * selection.sensitivities.size(),
        719U,
        1U,
    };
    const auto reference = price_gradient_test::execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        launch,
        cev::launch_cev_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto graph = run_graph<
        pg::SensitivityOrders::first_and_second,
        cev::price_gradients::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>
    >(
        plan,
        launch,
        {2U, 512U, 2U}
    );
    compare(reference, graph);
}

template<OptionSide Side>
void check_bates(std::size_t paths) {
    const std::vector<bates::ModelParameters> models{
        {1.05f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.3f, -0.7f,
         0.5f, -0.1f, 0.25f},
        {1.0f, -0.01f, 0.0f, 0.06f, 0.8f, 0.04f, 0.4f, -0.3f,
         5.0f, 0.02f, 0.4f},
        {100.0f, 0.08f, 0.02f, 0.0f, 1.0f, 0.04f, 0.3f, 1.0f,
         0.0f, -0.2f, 0.1f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 16U}, {1.1f, 12U}, {95.0f, 8U},
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f}},
        {"model.risk_free_rate", {
            .0005f, pg::BumpScale::absolute
        }},
        {"model.dividend_yield", {
            .0005f, pg::BumpScale::absolute
        }},
        {"model.initial_variance", {
            .001f, pg::BumpScale::absolute
        }},
        {"model.kappa", {.005f}},
        {"model.theta", {.005f}},
        {"model.gamma", {.005f}},
        {"model.rho", {.002f, pg::BumpScale::absolute}},
        {"model.jump_intensity", {
            .05f, pg::BumpScale::absolute
        }},
        {"model.jump_log_mean", {
            .002f, pg::BumpScale::absolute
        }},
        {"model.jump_log_volatility", {.005f}},
        {"product.strike", {.005f}},
        {"product.maturity_years", {
            1.f / 504.f, pg::BumpScale::absolute
        }},
    }};
    const auto plan = bates::prepare_bates_european_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {},
        selection,
        {pg::SensitivityOrders::first_and_second}
    );
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        models.size(),
        paths,
        128U,
        models.size() * selection.sensitivities.size(),
        1879U,
        1U,
    };
    const auto reference = price_gradient_test::execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        launch,
        bates::launch_bates_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto graph = run_graph<
        pg::SensitivityOrders::first_and_second,
        bates::price_gradients::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>,
        bates::EuropeanOptionPriceGradientPlan,
        13U,
        32U,
        2U
    >(
        plan,
        launch,
        {2U, 512U, 2U}
    );
    compare(reference, graph);
}

template<OptionSide Side>
void check_merton(std::size_t paths) {
    const std::vector<merton::ModelParameters> models{
        {1.05f, .03f, .01f, .2f, .5f, -.1f, .25f},
        {1.f, -.01f, 0.f, .35f, 25.f, .02f, .4f},
        {100.f, .08f, .02f, .1f, 0.f, -.2f, .1f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.f, 126U}, {1.1f, 126U}, {95.f, 504U},
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f}},
        {"model.risk_free_rate", {
            .0005f, pg::BumpScale::absolute
        }},
        {"model.dividend_yield", {
            .0005f, pg::BumpScale::absolute
        }},
        {"model.volatility", {.005f}},
        {"model.jump_intensity", {
            .05f, pg::BumpScale::absolute
        }},
        {"model.jump_log_mean", {
            .002f, pg::BumpScale::absolute
        }},
        {"model.jump_log_volatility", {.005f}},
        {"product.strike", {.005f}},
        {"product.maturity_years", {
            1.f / 504.f, pg::BumpScale::absolute
        }},
    }};
    const auto plan = merton::prepare_merton_european_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {},
        selection,
        {pg::SensitivityOrders::first_and_second}
    );
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        models.size(),
        paths,
        128U,
        models.size() * selection.sensitivities.size(),
        719U,
        1U,
    };
    const auto reference = price_gradient_test::execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        launch,
        merton::launch_merton_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto graph = run_graph<
        pg::SensitivityOrders::first_and_second,
        merton::price_gradients::CoupledDynamics,
        product::EuropeanOptionGradientPathPolicy<Side>,
        merton::EuropeanOptionPriceGradientPlan,
        9U,
        16U,
        2U
    >(
        plan,
        launch,
        {2U, 512U, 2U}
    );
    compare(reference, graph);
}

void check_active_workspace_capacity() {
    for (const std::size_t paths : {4096U, 1U << 20U}) {
        for (const std::size_t axes : {1U, 4U, 9U}) {
            const auto requirements =
                mcpg::terminal_node_graph_workspace_requirements<
                    pg::SensitivityOrders::first_and_second, 9U
                >(axes, 128U, {1U, paths, 1U});
            const auto nodes = 1U + 3U * axes;
            require(requirements.node_values == paths * nodes,
                    "Terminal graph reserves unused path nodes.");
            require(requirements.node_metadata == nodes,
                    "Terminal graph reserves unused metadata nodes.");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::size_t paths = 4097U;
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            paths = 513U;
        } else if (argc != 1) {
            throw std::invalid_argument("Usage: test [--sanitizer]");
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        check_active_workspace_capacity();
        check_heston<OptionSide::call>(
            paths, PriceConstruction::Aligned
        );
        check_heston<OptionSide::put>(
            paths, PriceConstruction::CartesianProduct
        );
        check_cev<OptionSide::put>(paths);
        check_merton<OptionSide::call>(paths);
        check_bates<OptionSide::put>(paths);
        std::cout
            << "Terminal node graph: Heston, CEV boundaries/time, Merton and "
               "Bates coupled jumps are bitwise equal to mono\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
