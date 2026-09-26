// Shared execution and parity checks for device-prepared American sensitivities.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "product/american_option/parameters.hpp"
#include "tests/price_gradients/cuda_test_support.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace price_gradient_test::american {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;
using ::ai_factory::workbench::PriceConstruction;

struct Results {
    std::vector<float> prices;
    std::vector<float> price_errors;
    std::vector<float> gradients;
    std::vector<float> gradient_errors;
    std::vector<float> diagonal_hessians;
    std::vector<float> diagonal_hessian_errors;
};

template<auto Launcher>
auto exact_transition_legacy_reference() {
    return [](
        const auto& models,
        const auto* device_models,
        const auto& products,
        const auto* device_products,
        pg::TimeConfiguration time,
        std::size_t paths,
        unsigned int threads,
        std::size_t blocks,
        std::uint64_t seed,
        float* outputs
    ) {
        const std::size_t rows = models.size();
        return Launcher(
            models.data(), device_models, models.size(),
            products.data(), device_products, products.size(),
            PriceConstruction::Aligned, rows, paths,
            time.dt * static_cast<float>(time.simulation_steps_per_day),
            threads, blocks, seed,
            ::ai_factory::workbench::equity::price_delta::
                SpotBumpConfiguration{0.01f},
            outputs, outputs + rows, outputs + 2U * rows,
            outputs + 3U * rows
        );
    };
}

template<auto Launcher>
auto fixed_step_legacy_reference() {
    return [](
        const auto& models,
        const auto* device_models,
        const auto& products,
        const auto* device_products,
        pg::TimeConfiguration time,
        std::size_t paths,
        unsigned int threads,
        std::size_t blocks,
        std::uint64_t seed,
        float* outputs
    ) {
        const std::size_t rows = models.size();
        return Launcher(
            models.data(), device_models, models.size(),
            products.data(), device_products, products.size(),
            PriceConstruction::Aligned, rows, paths,
            time.dt, time.simulation_steps_per_day,
            threads, blocks, seed,
            ::ai_factory::workbench::equity::price_delta::
                SpotBumpConfiguration{0.01f},
            outputs, outputs + rows, outputs + 2U * rows,
            outputs + 3U * rows
        );
    };
}

template<pg::SensitivityOrders Orders, typename Plan, typename Launcher>
Results execute(
    const Plan& plan,
    std::size_t paths,
    unsigned int threads,
    std::size_t blocks,
    std::uint64_t seed,
    Launcher&& launcher
) {
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    const std::size_t rows = plan.result_count;
    const std::size_t sensitivities = plan.sensitivity_count();

    DeviceArray<typename Plan::Preparation::Model> models(plan.models);
    DeviceArray<typename Plan::Preparation::Product> products(plan.products);
    DeviceArray<typename Plan::SensitivitySpec> specs(plan.sensitivities);
    DeviceArray<pg::SensitivityStencil<node_capacity>> stencils(
        rows * sensitivities
    );
    DeviceArray<
        ::ai_factory::workbench::equity::price_gradients::
            device_preparation::Error
    > preparation_error(1U);
    DeviceArray<float> prices_device(rows);
    DeviceArray<float> price_errors_device(rows);
    DeviceArray<float> gradients_device(
        pg::requests_first_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> gradient_errors_device(
        pg::requests_first_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> hessians_device(
        pg::requests_second_v<Orders> ? rows * sensitivities : 0U
    );
    DeviceArray<float> hessian_errors_device(
        pg::requests_second_v<Orders> ? rows * sensitivities : 0U
    );

    const typename Plan::DeviceInputs inputs{
        models.data,
        models.count,
        products.data,
        products.count,
        specs.data,
        specs.count,
    };
    const mcpg::DevicePreparedStencilOutputs<node_capacity>
        stencil_outputs{
            stencils.data,
            stencils.count,
            preparation_error.data,
        };
    const pg::SensitivityOutputs outputs{
        prices_device.data,
        price_errors_device.data,
        gradients_device.data,
        gradient_errors_device.data,
        hessians_device.data,
        hessian_errors_device.data,
        rows,
        rows * sensitivities,
    };
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        paths,
        threads,
        blocks,
        seed,
        1U,
    };
    const auto launched = launcher(
        plan, inputs, stencil_outputs, launch, outputs
    );
    ::ai_factory::workbench::longstaff_schwartz::
        validate_regression_diagnostics(
            launched, "exact-transition American sensitivity test"
        );
    require(
        preparation_error.read()[0U].code == 0,
        "American device sensitivity preparation failed."
    );
    return {
        prices_device.read(),
        price_errors_device.read(),
        gradients_device.read(),
        gradient_errors_device.read(),
        hessians_device.read(),
        hessian_errors_device.read(),
    };
}

inline void require_finite(const Results& result, const std::string& model) {
    const auto finite = [&](const std::vector<float>& values,
                            const char* quantity) {
        for (float value : values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    model + " American " + quantity + " is not finite."
                );
            }
        }
    };
    const auto valid_error = [&](const std::vector<float>& values,
                                 const char* quantity) {
        for (float value : values) {
            if (!std::isfinite(value) || value < 0.0f) {
                throw std::runtime_error(
                    model + " American " + quantity + " is invalid."
                );
            }
        }
    };
    finite(result.prices, "price");
    valid_error(result.price_errors, "price standard error");
    finite(result.gradients, "gradient");
    valid_error(result.gradient_errors, "gradient standard error");
    finite(result.diagonal_hessians, "diagonal Hessian");
    valid_error(
        result.diagonal_hessian_errors,
        "diagonal Hessian standard error"
    );
}

inline void require_close_relative(
    float actual,
    float expected,
    float tolerance,
    const char* message
) {
    const float scale = std::max(std::abs(expected), 1.0e-12f);
    if (std::abs(actual - expected) > tolerance * scale) {
        std::cerr << message << ": " << std::hexfloat << actual
                  << " vs " << expected << std::defaultfloat << '\n';
        throw std::runtime_error(message);
    }
}

template<
    typename Model,
    typename Prepare,
    typename FirstLauncher,
    typename DiagonalLauncher,
    typename NodeGraphLauncher,
    typename LegacyLauncher
>
void verify_model(
    const char* model_name,
    const std::vector<Model>& models,
    const std::vector<
        ::ai_factory::workbench::product::AmericanOptionParameters
    >& products,
    const pg::TimeConfiguration& time,
    const pg::PriceGradientConfiguration& full,
    std::size_t coupled_coordinate,
    std::size_t paths,
    Prepare&& prepare,
    FirstLauncher&& first_launcher,
    DiagonalLauncher&& diagonal_launcher,
    NodeGraphLauncher&& node_graph_launcher,
    LegacyLauncher&& legacy_launcher
) {
    constexpr unsigned int threads = 128U;
    constexpr std::size_t blocks = 4U;
    constexpr std::uint64_t seed = 91871U;
    require(!full.sensitivities.empty(), "Missing spot sensitivity.");
    require(
        coupled_coordinate < full.sensitivities.size(),
        "Invalid selected American sensitivity coordinate."
    );

    const auto first_plan = [&](const pg::PriceGradientConfiguration& c) {
        return prepare(c, pg::SensitivityOrders::first);
    };
    const auto price_only = execute<pg::SensitivityOrders::first>(
        first_plan({}), paths, threads, blocks, seed, first_launcher
    );
    const pg::PriceGradientConfiguration spot_configuration{{
        full.sensitivities.front()
    }};
    const auto spot_only = execute<pg::SensitivityOrders::first>(
        first_plan(spot_configuration),
        paths,
        threads,
        blocks,
        seed,
        first_launcher
    );
    const auto selected = execute<pg::SensitivityOrders::first>(
        first_plan(full), paths, threads, blocks, seed, first_launcher
    );

    DeviceArray<Model> device_models(models);
    DeviceArray<::ai_factory::workbench::product::AmericanOptionParameters>
        device_products(products);
    DeviceArray<float> legacy(4U * models.size());
    const auto legacy_launch = legacy_launcher(
        models,
        device_models.data,
        products,
        device_products.data,
        time,
        paths,
        threads,
        blocks,
        seed,
        legacy.data
    );
    ::ai_factory::workbench::longstaff_schwartz::
        validate_regression_diagnostics(
            legacy_launch, "exact-transition American price-delta reference"
        );
    const auto reference = legacy.read();
    const std::size_t sensitivity_count = full.sensitivities.size();
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(
            price_only.prices[row], spot_only.prices[row],
            "American empty selection changed the central price."
        );
        same(
            price_only.price_errors[row], spot_only.price_errors[row],
            "American empty selection changed the central error."
        );
        same(
            spot_only.prices[row], reference[row],
            "American central price differs from price-delta."
        );
        same(
            spot_only.price_errors[row], reference[models.size() + row],
            "American central error differs from price-delta."
        );
        require_close_relative(
            spot_only.gradients[row], reference[2U * models.size() + row],
            5.0e-6f,
            "American spot gradient differs from price-delta."
        );
        require_close_relative(
            spot_only.gradient_errors[row],
            reference[3U * models.size() + row],
            5.0e-5f,
            "American spot-gradient error differs from price-delta."
        );
        same(
            selected.prices[row], spot_only.prices[row],
            "Adding American sensitivities changed the central price."
        );
        same(
            selected.gradients[row * sensitivity_count],
            spot_only.gradients[row],
            "Adding American sensitivities changed the spot gradient."
        );
    }
    require_finite(selected, model_name);

    const pg::PriceGradientConfiguration one_configuration{{
        full.sensitivities[coupled_coordinate]
    }};
    const auto one = execute<pg::SensitivityOrders::first>(
        first_plan(one_configuration),
        paths,
        threads,
        blocks,
        seed,
        first_launcher
    );
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(
            one.gradients[row],
            selected.gradients[
                row * sensitivity_count + coupled_coordinate
            ],
            "American coordinate selection changed its gradient."
        );
        same(
            one.gradient_errors[row],
            selected.gradient_errors[
                row * sensitivity_count + coupled_coordinate
            ],
            "American coordinate selection changed its error."
        );
    }

    const auto diagonal_plan = prepare(
        full, pg::SensitivityOrders::first_and_second
    );
    const auto diagonal = execute<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan,
        paths,
        threads,
        blocks,
        seed,
        diagonal_launcher
    );
    require(
        diagonal.prices == selected.prices
            && diagonal.price_errors == selected.price_errors
            && diagonal.gradients == selected.gradients
            && diagonal.gradient_errors == selected.gradient_errors,
        "American diagonal request changed price or gradient outputs."
    );
    require_finite(diagonal, model_name);

    const auto node_graph = execute<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan,
        paths,
        threads,
        blocks,
        seed,
        node_graph_launcher
    );
    const auto require_same_values = [](
        const std::vector<float>& actual,
        const std::vector<float>& expected,
        const char* message
    ) {
        require(actual.size() == expected.size(), message);
        for (std::size_t index = 0U; index < actual.size(); ++index) {
            same(actual[index], expected[index], message);
        }
    };
    require_same_values(
        node_graph.prices, diagonal.prices,
        "American mono/node_graph prices differ."
    );
    require_same_values(
        node_graph.price_errors, diagonal.price_errors,
        "American mono/node_graph price errors differ."
    );
    require_same_values(
        node_graph.gradients, diagonal.gradients,
        "American mono/node_graph gradients differ."
    );
    require_same_values(
        node_graph.gradient_errors, diagonal.gradient_errors,
        "American mono/node_graph gradient errors differ."
    );
    require_same_values(
        node_graph.diagonal_hessians, diagonal.diagonal_hessians,
        "American mono/node_graph diagonal Hessians differ."
    );
    require_same_values(
        node_graph.diagonal_hessian_errors,
        diagonal.diagonal_hessian_errors,
        "American mono/node_graph diagonal Hessian errors differ."
    );
}

}  // namespace price_gradient_test::american
