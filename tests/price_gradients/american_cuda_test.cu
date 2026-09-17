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

template<OptionSide Side>
GradientRun execute(
    const heston::AmericanOptionPriceGradientPlan& plan,
    std::size_t paths,
    unsigned int threads,
    std::size_t blocks,
    std::uint64_t seed,
    bool split = false
) {
    const std::size_t rows = plan.result_count;
    const std::size_t sensitivities = plan.sensitivity_count();
    DeviceArray<heston::AmericanOptionPriceGradientPlan::ScenarioType>
        scenarios(plan.scenarios);
    DeviceArray<pg::Stencil> stencils(plan.stencils);
    DeviceArray<float> storage(2U * rows + 2U * rows * sensitivities);
    const pg::DeviceInputs<
        heston::AmericanOptionPriceGradientPlan::ScenarioType
    > inputs{
        scenarios.data,
        scenarios.count,
        stencils.data,
        stencils.count,
    };
    const pg::Outputs outputs{
        storage.data,
        storage.data + rows,
        storage.data + 2U * rows,
        storage.data + 2U * rows + rows * sensitivities,
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
    longstaff_schwartz::LaunchResult result;
    if (split) {
        launch.result_count = rows - 1U;
        result = heston::launch_heston_american_option_price_gradients_cuda<Side>(
            plan, inputs, launch, outputs
        );
        launch.result_offset = rows - 1U;
        launch.result_count = 1U;
        result = heston::launch_heston_american_option_price_gradients_cuda<Side>(
            plan, inputs, launch, outputs
        );
    } else {
        result = heston::launch_heston_american_option_price_gradients_cuda<Side>(
            plan, inputs, launch, outputs
        );
    }
    longstaff_schwartz::validate_regression_diagnostics(
        result, "Heston American price-gradients test"
    );
    const auto host = storage.read();
    return {
        {host.begin(), host.begin() + rows},
        {host.begin() + rows, host.begin() + 2U * rows},
        {host.begin() + 2U * rows,
         host.begin() + 2U * rows + rows * sensitivities},
        {host.begin() + 2U * rows + rows * sensitivities, host.end()},
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
    const auto time = equity::price_gradients::TimeConfiguration{
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
        const auto selected_spot = execute<Side>(
            spot_plan, paths, threads, blocks, seed
        );
        const auto price_only = execute<Side>(
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
        const auto extended = execute<Side>(
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

        const auto rho_only = execute<Side>(
            prepare({{full.sensitivities[7]}}),
            paths,
            threads,
            blocks,
            seed
        );
        const auto split = execute<Side>(
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
