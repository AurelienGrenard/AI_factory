// CIR Jamshidian compact first/diagonal benchmark for both work distributions.
#include "model/fixed_income/cir/product/european_swaption_price_gradients.cuh"
#include "tests/performance/benchmark_support.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

using namespace ai_factory::workbench;
namespace cir = model::fixed_income::cir;
namespace pg = price_gradients;
namespace mcpg = monte_carlo::price_gradients;
namespace perf = performance;

struct Run {
    perf::Measurement measurement;
    std::vector<float> values;
};

std::uint64_t output_hash(const std::vector<float>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite CIR benchmark output.");
        }
        hash ^= std::bit_cast<std::uint32_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

template<pg::SensitivityOrders Orders>
int run_benchmark(
    std::size_t rows,
    unsigned int scalar_threads,
    unsigned int cooperative_threads
) {
    constexpr bool second = pg::requests_second_v<Orders>;
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    std::vector<cir::ModelParameters> models;
    std::vector<product::RegularEuropeanSwaptionParameters> products;
    models.reserve(rows);
    products.reserve(rows);
    for (std::size_t row = 0U; row < rows; ++row) {
        models.push_back({
            {0.20f + 0.02f * static_cast<float>(row % 11U),
             0.03f + 0.002f * static_cast<float>(row % 5U),
             0.08f + 0.005f * static_cast<float>(row % 7U)},
            0.02f + 0.001f * static_cast<float>(row % 9U),
        });
        products.push_back({
            1.0f,
            0.025f + 0.001f * static_cast<float>(row % 17U),
            0.5f,
            252U,
            126U,
            4U + static_cast<std::uint32_t>(row % 9U),
        });
    }
    const pg::PriceGradientConfiguration selection{{
        {"model.mean_reversion", {.005}},
        {"model.long_term_mean", {.005}},
        {"model.volatility", {.005}},
        {"model.initial_state", {.0005, pg::BumpScale::absolute}},
        {"product.notional", {.005}},
        {"product.strike", {.0005, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {.005}},
    }};
    const auto plan = cir::prepare_cir_european_swaption_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        {Orders}
    );
    const std::size_t k = selection.sensitivities.size();
    const std::size_t sensitivity_count = rows * k;
    const std::size_t output_count = rows
        + sensitivity_count
        + (second ? sensitivity_count : 0U);

    perf::DeviceBuffer model_buffer(
        plan.models.size() * sizeof(cir::ModelParameters),
        perf::DeviceMemoryRole::persistent_input
    );
    perf::DeviceBuffer product_buffer(
        plan.products.size()
            * sizeof(product::RegularEuropeanSwaptionParameters),
        perf::DeviceMemoryRole::persistent_input
    );
    perf::DeviceBuffer sensitivity_buffer(
        plan.sensitivities.size()
            * sizeof(cir::EuropeanSwaptionPriceGradientPlan::SensitivitySpec),
        perf::DeviceMemoryRole::persistent_input
    );
    perf::DeviceBuffer stencil_buffer(
        sensitivity_count * sizeof(pg::SensitivityStencil<node_capacity>),
        perf::DeviceMemoryRole::output
    );
    perf::DeviceBuffer error_buffer(
        sizeof(pg::device_preparation::Error),
        perf::DeviceMemoryRole::output
    );
    perf::DeviceBuffer output_buffer(
        output_count * sizeof(float),
        perf::DeviceMemoryRole::output
    );
    perf::copy_to_device(model_buffer, plan.models);
    perf::copy_to_device(product_buffer, plan.products);
    perf::copy_to_device(sensitivity_buffer, plan.sensitivities);
    check_cuda(
        cudaMemset(error_buffer.as<void>(), 0, sizeof(pg::device_preparation::Error)),
        "CIR benchmark reset preparation error"
    );

    const cir::EuropeanSwaptionPriceGradientPlan::DeviceInputs inputs{
        model_buffer.as<cir::ModelParameters>(), plan.models.size(),
        product_buffer.as<product::RegularEuropeanSwaptionParameters>(),
        plan.products.size(),
        sensitivity_buffer.as<
            cir::EuropeanSwaptionPriceGradientPlan::SensitivitySpec
        >(),
        plan.sensitivities.size(),
    };
    const mcpg::DevicePreparedStencilOutputs<node_capacity> stencil_outputs{
        stencil_buffer.as<pg::SensitivityStencil<node_capacity>>(),
        sensitivity_count,
        error_buffer.as<pg::device_preparation::Error>(),
    };
    const pg::SensitivityOutputs outputs{
        output_buffer.as<float>(),
        nullptr,
        output_buffer.as<float>() + rows,
        nullptr,
        second
            ? output_buffer.as<float>() + rows + sensitivity_count
            : nullptr,
        nullptr,
        rows,
        sensitivity_count,
    };

    const auto run = [&] (
        closed_form::WorkDistribution distribution,
        unsigned int threads
    ) {
        const std::size_t blocks = distribution
                == closed_form::WorkDistribution::cooperative
            ? rows
            : (rows + threads - 1U) / threads;
        const pg::LaunchConfiguration configuration{
            pg::PricingMethod::closed_form,
            0U,
            rows,
            0U,
            threads,
            blocks,
            0U,
            1U,
        };
        const auto invoke = [&] {
            if constexpr (second) {
                cir::launch_cir_european_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    Orders
                >(
                    plan,
                    inputs,
                    stencil_outputs,
                    configuration,
                    outputs,
                    distribution
                );
            } else {
                cir::launch_cir_european_swaption_price_gradients_cuda<
                    SwaptionSide::payer
                >(
                    plan,
                    inputs,
                    stencil_outputs,
                    configuration,
                    pg::Outputs{
                        outputs.prices,
                        nullptr,
                        outputs.gradients,
                        nullptr,
                        outputs.price_capacity,
                        outputs.sensitivity_capacity,
                    },
                    distribution
                );
            }
        };
        const auto measurement = perf::measure_cuda(invoke);
        invoke();
        check_cuda(cudaDeviceSynchronize(), "CIR benchmark synchronize");
        return Run{
            measurement,
            perf::copy_from_device<float>(output_buffer, output_count),
        };
    };

    const auto scalar = run(
        closed_form::WorkDistribution::scalar, scalar_threads
    );
    const auto cooperative = run(
        closed_form::WorkDistribution::cooperative, cooperative_threads
    );
    const auto stencils = perf::copy_from_device<
        pg::SensitivityStencil<node_capacity>
    >(stencil_buffer, sensitivity_count);
    const auto preparation_error = perf::copy_from_device<
        pg::device_preparation::Error
    >(error_buffer, 1U)[0U];
    if (preparation_error.code != pg::device_preparation::valid) {
        throw std::runtime_error("CIR benchmark sensitivity preparation failed.");
    }

    double maximum_difference = 0.0;
    double maximum_scaled_price_difference = 0.0;
    double maximum_scaled_gradient_difference = 0.0;
    double maximum_gradient_tolerance_ratio = 0.0;
    double maximum_hessian_difference = 0.0;
    for (std::size_t index = 0U; index < output_count; ++index) {
        const double difference = std::abs(
            static_cast<double>(scalar.values[index])
                - static_cast<double>(cooperative.values[index])
        );
        maximum_difference = std::max(maximum_difference, difference);
        const double scale = std::max(
            1.0,
            std::abs(static_cast<double>(scalar.values[index]))
        );
        if (index < rows) {
            maximum_scaled_price_difference = std::max(
                maximum_scaled_price_difference,
                difference / scale
            );
        } else if (index < rows + sensitivity_count) {
            maximum_scaled_gradient_difference = std::max(
                maximum_scaled_gradient_difference,
                difference / scale
            );
            const auto& stencil = stencils[index - rows];
            constexpr double scenario_price_tolerance = 3.0e-5;
            const double cancellation_tolerance =
                stencil.kind == pg::StencilKind::centered
                ? 2.0 * scenario_price_tolerance
                    / std::abs(static_cast<double>(
                        stencil.represented_width
                    ))
                : 2.0 * scenario_price_tolerance
                    * (std::abs(static_cast<double>(
                           stencil.first_endpoint_weights[0U]
                       ))
                       + std::abs(static_cast<double>(
                           stencil.first_endpoint_weights[1U]
                       )));
            const double allowed_difference = std::max(
                2.0e-3 * scale,
                cancellation_tolerance
            );
            maximum_gradient_tolerance_ratio = std::max(
                maximum_gradient_tolerance_ratio,
                difference / allowed_difference
            );
        } else {
            maximum_hessian_difference = std::max(
                maximum_hessian_difference,
                difference
            );
        }
    }
    if (maximum_scaled_price_difference > 3.0e-5
        || maximum_gradient_tolerance_ratio > 1.0) {
        throw std::runtime_error(
            "CIR scalar/cooperative price-gradient tolerance exceeded."
        );
    }

    const auto emit = [&] (
        const char* distribution_name,
        const Run& result,
        unsigned int threads,
        std::size_t blocks
    ) {
        perf::emit_measurement(
            "PERF-023",
            "price_gradients_cir_jamshidian",
            std::string(distribution_name)
                + (second ? "/first_and_diagonal_second" : "/first"),
            result.measurement,
            {
                {"rows", rows},
                {"sensitivity_count", k},
                {"node_capacity", node_capacity},
                {"threads_per_block", threads},
                {"block_count", blocks},
                {"maximum_payment_count", plan.maximum_payment_count},
            },
            {
                {"finite", true},
                {"output_hash", output_hash(result.values)},
                {"scalar_cooperative_maximum_absolute_difference",
                 maximum_difference},
                {"scalar_cooperative_maximum_scaled_price_difference",
                 maximum_scaled_price_difference},
                {"scalar_cooperative_maximum_scaled_gradient_difference",
                 maximum_scaled_gradient_difference},
                {"scalar_cooperative_maximum_gradient_tolerance_ratio",
                 maximum_gradient_tolerance_ratio},
                {"scalar_cooperative_maximum_hessian_difference",
                 maximum_hessian_difference},
                {"scalar_cooperative_price_gradient_within_tolerance", true},
            }
        );
    };
    emit(
        "scalar",
        scalar,
        scalar_threads,
        (rows + scalar_threads - 1U) / scalar_threads
    );
    emit("cooperative", cooperative, cooperative_threads, rows);
    return 0;
}

int main(int argc, char** argv) {
    try {
        const std::size_t rows = argc > 1 ? std::stoull(argv[1]) : 1000U;
        const unsigned int scalar_threads =
            argc > 2 ? std::stoul(argv[2]) : 256U;
        const unsigned int cooperative_threads =
            argc > 3 ? std::stoul(argv[3]) : 128U;
        const std::string_view mode = argc > 4 ? argv[4] : "first";
        if (argc > 5 || rows == 0U
            || (mode != "first" && mode != "diagonal")) {
            throw std::invalid_argument(
                "Usage: benchmark_price_gradients_jamshidian "
                "[ROWS [SCALAR_THREADS [COOPERATIVE_THREADS "
                "[first|diagonal]]]]"
            );
        }
        if (mode == "diagonal") {
            return run_benchmark<
                pg::SensitivityOrders::first_and_second
            >(rows, scalar_threads, cooperative_threads);
        }
        return run_benchmark<pg::SensitivityOrders::first>(
            rows, scalar_threads, cooperative_threads
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
