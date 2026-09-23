// Heston American sensitivity pipeline benchmark with per-phase diagnostics.
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"
#include "tests/performance/benchmark_support.cuh"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

using namespace ai_factory::workbench;
namespace heston = model::equity::heston;
namespace pg = price_gradients;
namespace perf = performance;

// The multi-kernel LSM pipeline needs a longer device warmup than an isolated
// kernel before clocks and allocator state reach the measured steady state.
inline constexpr int kPipelineWarmups = 20;
inline constexpr std::size_t kPipelinesPerSample = 4U;

std::uint64_t output_hash(const std::vector<float>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Non-finite Heston American benchmark output."
            );
        }
        hash ^= std::bit_cast<std::uint32_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

template<pg::SensitivityOrders Orders>
int run_benchmark(
    std::size_t paths,
    std::size_t rows,
    unsigned int threads,
    std::size_t blocks
) {
    static_assert(
        Orders == pg::SensitivityOrders::first
        || Orders == pg::SensitivityOrders::first_and_second
    );
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    constexpr bool diagonal = pg::requests_second_v<Orders>;
    std::vector<heston::ModelParameters> models;
    std::vector<product::AmericanOptionParameters> products;
    models.reserve(rows);
    products.reserve(rows);
    for (std::size_t row = 0U; row < rows; ++row) {
        models.push_back({
            0.90f + 0.05f * static_cast<float>(row % 5U),
            0.03f,
            0.01f,
            0.04f,
            1.5f,
            0.04f,
            0.40f,
            -0.70f,
        });
        products.push_back({1.0f, 126U, 7U});
    }
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005}},
        {"model.initial_variance", {.001, pg::BumpScale::absolute}},
        {"model.risk_free_rate", {.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {.0005, pg::BumpScale::absolute}},
        {"model.kappa", {.005}},
        {"model.theta", {.005}},
        {"model.gamma", {.005}},
        {"model.rho", {.002, pg::BumpScale::absolute}},
        {"product.strike", {.005}},
    }};
    const auto plan = heston::prepare_heston_american_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        {Orders}
    );
    using Plan = heston::AmericanOptionPriceGradientPlan;
    perf::DeviceBuffer model_buffer(
        plan.models.size() * sizeof(Plan::Model),
        perf::DeviceMemoryRole::persistent_input
    );
    perf::DeviceBuffer product_buffer(
        plan.products.size() * sizeof(Plan::Product),
        perf::DeviceMemoryRole::persistent_input
    );
    perf::DeviceBuffer sensitivity_buffer(
        plan.sensitivities.size() * sizeof(Plan::SensitivitySpec),
        perf::DeviceMemoryRole::persistent_input
    );
    const std::size_t sensitivity_count = selection.sensitivities.size();
    const std::size_t sensitivity_value_count = rows * sensitivity_count;
    perf::DeviceBuffer stencil_buffer(
        sensitivity_value_count
            * sizeof(pg::SensitivityStencil<node_capacity>),
        perf::DeviceMemoryRole::output
    );
    perf::DeviceBuffer preparation_error_buffer(
        sizeof(equity::price_gradients::device_preparation::Error),
        perf::DeviceMemoryRole::output
    );
    const std::size_t output_count = 2U * rows
        + (diagonal ? 4U : 2U) * sensitivity_value_count;
    perf::DeviceBuffer output_buffer(
        output_count * sizeof(float),
        perf::DeviceMemoryRole::output
    );
    perf::copy_to_device(model_buffer, plan.models);
    perf::copy_to_device(product_buffer, plan.products);
    perf::copy_to_device(sensitivity_buffer, plan.sensitivities);
    const Plan::DeviceInputs inputs{
        model_buffer.as<Plan::Model>(),
        plan.models.size(),
        product_buffer.as<Plan::Product>(),
        plan.products.size(),
        sensitivity_buffer.as<Plan::SensitivitySpec>(),
        plan.sensitivities.size(),
    };
    const monte_carlo::price_gradients::DevicePreparedStencilOutputs<
        node_capacity
    > stencil_outputs{
        stencil_buffer.as<pg::SensitivityStencil<node_capacity>>(),
        sensitivity_value_count,
        preparation_error_buffer.as<
            equity::price_gradients::device_preparation::Error
        >(),
    };
    float* output = output_buffer.as<float>();
    float* const gradient = output + 2U * rows;
    float* const gradient_error = gradient + sensitivity_value_count;
    float* const hessian = diagonal
        ? gradient_error + sensitivity_value_count
        : nullptr;
    float* const hessian_error = diagonal
        ? hessian + sensitivity_value_count
        : nullptr;
    const pg::SensitivityOutputs outputs{
        output,
        output + rows,
        gradient,
        gradient_error,
        hessian,
        hessian_error,
        rows,
        sensitivity_value_count,
    };
    const pg::LaunchConfiguration configuration{
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        paths,
        threads,
        blocks,
        41871U,
        1U,
    };
    longstaff_schwartz::LaunchResult last;
    const auto invoke = [&]() {
        if constexpr (diagonal) {
            last = heston::
                launch_heston_american_option_diagonal_sensitivities_cuda<
                    OptionSide::call,
                    Orders
                >(
                    plan,
                    inputs,
                    stencil_outputs,
                    configuration,
                    outputs
                );
        } else {
            last = heston::launch_heston_american_option_price_gradients_cuda<
                OptionSide::call
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
        }
        longstaff_schwartz::validate_regression_diagnostics(
            last,
            "Heston American sensitivity performance benchmark"
        );
        return last.kernel_seconds * 1000.0;
    };
    const auto measurement = perf::measure_synchronous_cuda_pipeline(
        invoke,
        kPipelineWarmups,
        perf::kDefaultRepetitions,
        kPipelinesPerSample
    );
    invoke();
    const auto host = perf::copy_from_device<float>(
        output_buffer,
        output_count
    );
    perf::emit_measurement(
        "PERF-023",
        "price_gradients_heston_american",
        diagonal
            ? "frozen_exercise/first_and_diagonal_second/B=1"
            : "frozen_exercise/first/B=1",
        measurement,
        {
            {"rows", rows},
            {"sensitivity_count", sensitivity_count},
            {"paths_per_price", paths},
            {"threads_per_block", threads},
            {"blocks_per_price", blocks},
            {"node_capacity", node_capacity},
            {"seed", configuration.base_seed},
            {"batch_count", last.batch_count},
            {"kernel_launch_count", last.kernel_launch_count},
            {"maximum_prices_per_batch", last.maximum_prices_per_batch},
        },
        {{"finite", true}, {"output_hash", output_hash(host)}},
        kPipelineWarmups,
        perf::kDefaultRepetitions,
        {},
        last.workspace_bytes
    );
    return 0;
}

int main(int argc, char** argv) {
    try {
        const std::size_t paths = argc > 1 ? std::stoull(argv[1]) : 262144U;
        const std::size_t rows = argc > 2 ? std::stoull(argv[2]) : 4U;
        const unsigned int threads = argc > 3 ? std::stoul(argv[3]) : 256U;
        const std::size_t blocks = argc > 4 ? std::stoull(argv[4]) : 128U;
        const std::string_view order = argc > 5 ? argv[5] : "first";
        if (argc > 6 || rows == 0U
            || (order != "first" && order != "diagonal")) {
            throw std::invalid_argument(
                "Usage: benchmark_price_gradients_american "
                "[PATHS [ROWS [THREADS [BLOCKS [first|diagonal]]]]]"
            );
        }
        if (order == "diagonal") {
            return run_benchmark<
                pg::SensitivityOrders::first_and_second
            >(paths, rows, threads, blocks);
        }
        return run_benchmark<pg::SensitivityOrders::first>(
            paths, rows, threads, blocks
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
