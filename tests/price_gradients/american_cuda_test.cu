// Heston American selected gradients: central/delta parity and frozen replay.
#include "model/equity/markovian/heston/product/american_option.cuh"
#include "model/equity/markovian/heston/product/american_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"
#include "tests/price_gradients/cuda_test_support.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace heston = model::equity::heston;
namespace pg = price_gradients;
using price_gradient_test::DeviceArray;
using price_gradient_test::require;
using price_gradient_test::same;

std::size_t test_paths = 4096U;

struct GradientRun {
    std::vector<float> prices;
    std::vector<float> price_errors;
    std::vector<float> gradients;
    std::vector<float> gradient_errors;
    std::vector<float> diagonal_hessians;
    std::vector<float> diagonal_hessian_errors;
    std::vector<pg::SensitivityStencil<4U>> diagonal_stencils;
    longstaff_schwartz::LaunchResult launch;
};

void close_relative(
    float actual,
    float expected,
    float relative_tolerance,
    const char* message
) {
    const float scale = std::max(std::abs(expected), 1.0e-12f);
    if (!(std::abs(actual - expected) <= relative_tolerance * scale)) {
        std::cerr << message << ": " << std::hexfloat << actual
                  << " vs " << expected << std::defaultfloat << '\n';
        throw std::runtime_error(message);
    }
}

template<pg::SensitivityOrders Orders, OptionSide Side>
GradientRun execute(
    const heston::AmericanOptionPriceGradientPlan& plan,
    std::size_t paths,
    unsigned int threads,
    std::size_t blocks,
    std::uint64_t seed,
    bool split = false,
    bool node_graph = false
) {
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    const std::size_t rows = plan.result_count;
    const std::size_t sensitivities = plan.sensitivity_count();
    DeviceArray<heston::AmericanOptionPriceGradientPlan::Model>
        models(plan.models);
    DeviceArray<heston::AmericanOptionPriceGradientPlan::Product>
        products(plan.products);
    DeviceArray<heston::AmericanOptionPriceGradientPlan::SensitivitySpec>
        sensitivity_specs(plan.sensitivities);
    DeviceArray<pg::SensitivityStencil<node_capacity>> stencils(
        rows * sensitivities
    );
    DeviceArray<equity::price_gradients::device_preparation::Error> error(1U);
    DeviceArray<float> prices(rows);
    DeviceArray<float> price_errors(rows);
    DeviceArray<float> gradients(
        pg::requests_first_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> gradient_errors(
        pg::requests_first_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> hessians(
        pg::requests_second_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> hessian_errors(
        pg::requests_second_v<Orders> ? rows * sensitivities : 0U
    );
    const heston::AmericanOptionPriceGradientPlan::DeviceInputs inputs{
        models.data,
        models.count,
        products.data,
        products.count,
        sensitivity_specs.data,
        sensitivity_specs.count,
    };
    const monte_carlo::price_gradients::DevicePreparedStencilOutputs<
        node_capacity
    > stencil_outputs{
        stencils.data,
        stencils.count,
        error.data,
    };
    const pg::SensitivityOutputs outputs{
        prices.data,
        price_errors.data,
        gradients.data,
        gradient_errors.data,
        hessians.data,
        hessian_errors.data,
        rows,
        rows * sensitivities,
    };
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        paths,
        threads,
        blocks,
        seed,
        1U,
    };
    std::size_t node_graph_workspace_bytes = 0U;
    if constexpr (pg::requests_second_v<Orders>) {
        if (node_graph) {
            node_graph_workspace_bytes =
                heston::heston_american_option_node_graph_workspace_bytes<
                    Side, Orders
                >(plan, launch);
        }
    }
    DeviceArray<std::uint8_t> node_graph_workspace(
        node_graph_workspace_bytes
    );
    const auto invoke = [&](const auto& configuration) {
        if constexpr (Orders == pg::SensitivityOrders::first) {
            return heston::launch_heston_american_option_price_gradients_cuda<
                Side
            >(
                plan,
                inputs,
                stencil_outputs,
                configuration,
                {
                    outputs.prices,
                    outputs.price_standard_errors,
                    outputs.gradients,
                    outputs.gradient_standard_errors,
                    outputs.price_capacity,
                    outputs.sensitivity_capacity,
                }
            );
        } else {
            if (node_graph) {
                return heston::
                    launch_heston_american_option_node_graph_sensitivities_cuda<
                        Side,
                        Orders
                    >(
                        plan,
                        inputs,
                        stencil_outputs,
                        configuration,
                        outputs,
                        node_graph_workspace.data,
                        node_graph_workspace.count
                    );
            }
            return heston::
                launch_heston_american_option_diagonal_sensitivities_cuda<
                    Side,
                    Orders
                >(
                    plan,
                    inputs,
                    stencil_outputs,
                    configuration,
                    outputs
                );
        }
    };
    longstaff_schwartz::LaunchResult result;
    if (split) {
        launch.result_count = rows - 1U;
        result = invoke(launch);
        launch.result_offset = rows - 1U;
        launch.result_count = 1U;
        result = invoke(launch);
    } else {
        result = invoke(launch);
    }
    longstaff_schwartz::validate_regression_diagnostics(
        result, "Heston American price-gradients test"
    );
    require(
        error.read()[0U].code == 0,
        "American device sensitivity preparation failed."
    );
    std::vector<pg::SensitivityStencil<4U>> diagonal_stencils;
    if constexpr (node_capacity == 4U) {
        diagonal_stencils = stencils.read();
    }
    return {
        prices.read(),
        price_errors.read(),
        gradients.read(),
        gradient_errors.read(),
        hessians.read(),
        hessian_errors.read(),
        std::move(diagonal_stencils),
        result,
    };
}

template<OptionSide Side>
void run() {
    constexpr std::size_t rows = 3U;
    const std::size_t paths = test_paths;
    constexpr std::size_t blocks = 8U;
    constexpr std::uint64_t seed = 41871U;
    const std::vector<heston::ModelParameters> models{
        {1.00f, .03f, .01f, .04f, 1.5f, .04f, .40f, -.7f},
        {.15f, .20f, .00f, .06f, 1.1f, .05f, .35f, -.4f},
        {1.20f, .00f, .00f, .00f, .8f, .04f, .30f, 1.0f},
    };
    const std::vector<product::AmericanOptionParameters> products{
        {1.00f, 63U, 7U},
        {1.00f, 21U, 7U},
        {1.10f, 20U, 7U},
    };
    const auto time = pg::TimeConfiguration{
        1.0f / 504.0f, 2U
    };
    const pg::Sensitivity spot{"model.spot", {.005}};
    const auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return heston::prepare_heston_american_option_price_gradients(
            models,
            products,
            PriceConstruction::Aligned,
            time,
            selection
        );
    };

    DeviceArray<heston::ModelParameters> device_models(models);
    DeviceArray<product::AmericanOptionParameters> device_products(products);
    DeviceArray<float> legacy(4U * rows);

    for (unsigned int threads : {128U, 256U}) {
        const auto spot_plan = prepare({{spot}});
        const auto selected_spot = execute<pg::SensitivityOrders::first, Side>(
            spot_plan, paths, threads, blocks, seed
        );
        const auto price_only = execute<pg::SensitivityOrders::first, Side>(
            prepare({}), paths, threads, blocks, seed
        );
        const auto delta_launch =
            heston::launch_heston_american_option_price_delta_cuda<Side>(
                models.data(),
                device_models.data,
                rows,
                products.data(),
                device_products.data,
                rows,
                PriceConstruction::Aligned,
                rows,
                paths,
                time.dt,
                time.simulation_steps_per_day,
                threads,
                blocks,
                seed,
                equity::price_delta::SpotBumpConfiguration{.01f},
                legacy.data,
                legacy.data + rows,
                legacy.data + 2U * rows,
                legacy.data + 3U * rows
            );
        longstaff_schwartz::validate_regression_diagnostics(
            delta_launch, "Heston American price-delta reference"
        );
        const auto reference = legacy.read();
        for (std::size_t row = 0U; row < rows; ++row) {
            same(price_only.prices[row], selected_spot.prices[row],
                 "Empty American selection changed central price");
            same(price_only.price_errors[row], selected_spot.price_errors[row],
                 "Empty American selection changed central error");
            same(selected_spot.prices[row], reference[row],
                 "American central price differs from price-delta");
            same(selected_spot.price_errors[row], reference[rows + row],
                 "American central error differs from price-delta");
            close_relative(
                selected_spot.gradients[row],
                reference[2U * rows + row],
                5.0e-6f,
                "American selected spot gradient differs from price-delta"
            );
            close_relative(
                selected_spot.gradient_errors[row],
                reference[3U * rows + row],
                5.0e-5f,
                "American selected spot error differs from price-delta"
            );
        }
        require(
            selected_spot.launch.kernel_launch_count
                == delta_launch.kernel_launch_count,
            "A selected American gradient must add exactly two kernels."
        );
        require(
            selected_spot.launch.kernel_launch_count
                == price_only.launch.kernel_launch_count
                    + 2U * price_only.launch.batch_count,
            "American selected gradients must add two kernels per LSM batch."
        );

        const pg::PriceGradientConfiguration full{{
            spot,
            {"model.initial_variance", {.001, pg::BumpScale::absolute}},
            {"model.risk_free_rate", {.0005, pg::BumpScale::absolute}},
            {"model.dividend_yield", {.0005, pg::BumpScale::absolute}},
            {"model.kappa", {.005}},
            {"model.theta", {.005}},
            {"model.gamma", {.005}},
            {"model.rho", {.002, pg::BumpScale::absolute}},
            {"product.strike", {.005}},
        }};
        const auto extended = execute<pg::SensitivityOrders::first, Side>(
            prepare(full), paths, threads, blocks, seed
        );
        for (std::size_t row = 0U; row < rows; ++row) {
            same(extended.prices[row], selected_spot.prices[row],
                 "Adding American sensitivities changed central price");
            same(extended.price_errors[row], selected_spot.price_errors[row],
                 "Adding American sensitivities changed central error");
            same(extended.gradients[row * full.sensitivities.size()],
                 selected_spot.gradients[row],
                 "Adding American sensitivities changed spot gradient");
            same(extended.gradient_errors[row * full.sensitivities.size()],
                 selected_spot.gradient_errors[row],
                 "Adding American sensitivities changed spot error");
        }
        for (float value : extended.gradients) {
            require(std::isfinite(value),
                    "American frozen gradient is not finite.");
        }
        for (float value : extended.gradient_errors) {
            require(std::isfinite(value) && value >= 0.0f,
                    "American frozen gradient error is invalid.");
        }

        const auto rho_only = execute<pg::SensitivityOrders::first, Side>(
            prepare({{full.sensitivities[7]}}),
            paths,
            threads,
            blocks,
            seed
        );
        const auto split = execute<pg::SensitivityOrders::first, Side>(
            prepare(full), paths, threads, blocks, seed, true
        );
        require(split.prices == extended.prices
                && split.price_errors == extended.price_errors
                && split.gradients == extended.gradients
                && split.gradient_errors == extended.gradient_errors,
                "American result-offset batching changed outputs.");
        for (std::size_t row = 0U; row < rows; ++row) {
            same(rho_only.gradients[row],
                 extended.gradients[row * full.sensitivities.size() + 7U],
                 "American sensitivity selection changed rho gradient");
            same(rho_only.gradient_errors[row],
                 extended.gradient_errors[
                     row * full.sensitivities.size() + 7U
                 ],
                 "American sensitivity selection changed rho error");
        }

        const auto diagonal_plan =
            heston::prepare_heston_american_option_sensitivities(
                models,
                products,
                PriceConstruction::Aligned,
                time,
                full,
                {pg::SensitivityOrders::first_and_second}
            );
        const auto diagonal = execute<
            pg::SensitivityOrders::first_and_second,
            Side
        >(diagonal_plan, paths, threads, blocks, seed);
        const auto node_graph_diagonal = execute<
            pg::SensitivityOrders::first_and_second,
            Side
        >(diagonal_plan, paths, threads, blocks, seed, false, true);
        require(
            node_graph_diagonal.prices == diagonal.prices
                && node_graph_diagonal.price_errors == diagonal.price_errors
                && node_graph_diagonal.gradients == diagonal.gradients
                && node_graph_diagonal.gradient_errors
                    == diagonal.gradient_errors
                && node_graph_diagonal.diagonal_hessians
                    == diagonal.diagonal_hessians
                && node_graph_diagonal.diagonal_hessian_errors
                    == diagonal.diagonal_hessian_errors,
            "American mono and node_graph sensitivities differ."
        );
        require(
            diagonal.prices == extended.prices
                && diagonal.price_errors == extended.price_errors,
            "Adding American diagonal Hessians changed the central price."
        );
        for (std::size_t value = 0U;
             value < diagonal.gradients.size();
             ++value) {
            same(
                diagonal.gradients[value],
                extended.gradients[value],
                "Adding American diagonal Hessians changed a gradient"
            );
            same(
                diagonal.gradient_errors[value],
                extended.gradient_errors[value],
                "Adding American diagonal Hessians changed a gradient error"
            );
        }
        for (float value : diagonal.diagonal_hessians) {
            require(std::isfinite(value),
                    "American diagonal Hessian is not finite.");
        }
        for (float value : diagonal.diagonal_hessian_errors) {
            require(std::isfinite(value) && value >= 0.0f,
                    "American diagonal Hessian error is invalid.");
        }
        require(
            diagonal.diagonal_stencils[
                2U * full.sensitivities.size() + 1U
            ].node_count == 4U,
            "The variance boundary did not use four one-sided nodes."
        );

        if constexpr (Side == OptionSide::put) {
            require(selected_spot.prices[1] == 1.0f - models[1].spot,
                    "American put fixture must exercise initially.");
            require(selected_spot.price_errors[1] == 0.0f
                    && selected_spot.gradient_errors[1] == 0.0f,
                    "Initial exercise gradient must be deterministic.");
            const std::size_t first = full.sensitivities.size();
            require(extended.gradients[first] == -1.0f
                    && extended.gradients[
                        first + full.sensitivities.size() - 1U
                    ] == 1.0f,
                    "Initial exercise used dynamics instead of time-zero payoff.");
            for (std::size_t sensitivity = 1U;
                 sensitivity + 1U < full.sensitivities.size();
                 ++sensitivity) {
                require(extended.gradients[first + sensitivity] == 0.0f,
                        "Initial exercise retained a dynamics sensitivity.");
            }
            for (std::size_t sensitivity = 0U;
                 sensitivity < full.sensitivities.size();
                 ++sensitivity) {
                require(extended.gradient_errors[first + sensitivity] == 0.0f,
                        "Initial exercise gradient error must be zero.");
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            test_paths = 257U;
        } else if (argc != 1) {
            throw std::invalid_argument(
                "Usage: test_price_gradients_american_cuda [--sanitizer]"
            );
        }
        run<OptionSide::call>();
        run<OptionSide::put>();
        std::cout << "Heston American frozen selected gradients passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
