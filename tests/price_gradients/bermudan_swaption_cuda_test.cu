// Frozen-policy Bermudan sensitivities across fixed-income pricing paths.
#include "common/longstaff_schwartz/launch.cuh"
#include "model/fixed_income/cir/product/bermudan_swaption.cuh"
#include "model/fixed_income/cir/product/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/cir_plus_plus/product/flat/bermudan_swaption.cuh"
#include "model/fixed_income/cir_plus_plus/product/flat/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/cir_plus_plus/product/nelson_siegel/bermudan_swaption.cuh"
#include "model/fixed_income/cir_plus_plus/product/nelson_siegel/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/g2/product/bermudan_swaption.cuh"
#include "model/fixed_income/g2/product/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/product/svensson/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/vasicek/product/bermudan_swaption.cuh"
#include "model/fixed_income/vasicek/product/bermudan_swaption_price_gradients.cuh"
#include "tests/price_gradients/cuda_test_support.cuh"
#include "tests/price_gradients/mixed_node_graph_cuda_test_support.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ai_factory::workbench;
using price_gradient_test::DeviceArray;
using price_gradient_test::require;
using price_gradient_test::same;
namespace pg = price_gradients;
namespace prep_error = equity::price_gradients::device_preparation;

constexpr std::size_t kPaths = 4'096U;
constexpr unsigned int kThreads = 128U;
constexpr std::size_t kBlocks = 4U;
constexpr std::uint64_t kSeed = 2'170'000'019ULL;
constexpr pg::TimeConfiguration kTime{1.0f / 504.0f, 2U};

struct Results {
    std::vector<float> prices;
    std::vector<float> price_errors;
    std::vector<float> gradients;
    std::vector<float> gradient_errors;
    std::vector<float> hessians;
    std::vector<float> hessian_errors;
};

const std::vector<product::BermudanSwaptionParameters> kProducts{
    {1.0f, 0.030f, 0.5f, 126U, 126U, 6U, 4U},
    {0.8f, 0.040f, 0.25f, 252U, 63U, 8U, 5U},
};

pg::LaunchConfiguration launch_configuration(std::size_t rows) {
    return {
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        kPaths,
        kThreads,
        kBlocks,
        kSeed,
        1U,
    };
}

template<pg::SensitivityOrders Orders>
void require_valid(const Results& result, const char* name) {
    const auto finite = [&](const std::vector<float>& values,
                            const char* quantity) {
        for (float value : values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    std::string{name} + " has a non-finite " + quantity
                );
            }
        }
    };
    const auto valid_errors = [&](const std::vector<float>& values,
                                  const char* quantity) {
        finite(values, quantity);
        for (float value : values) {
            require(value >= 0.0f, "Negative Bermudan standard error.");
        }
    };
    finite(result.prices, "price");
    valid_errors(result.price_errors, "price standard error");
    if constexpr (pg::requests_first_v<Orders>) {
        finite(result.gradients, "gradient");
        valid_errors(result.gradient_errors, "gradient standard error");
    }
    if constexpr (pg::requests_second_v<Orders>) {
        finite(result.hessians, "diagonal Hessian");
        valid_errors(
            result.hessian_errors, "diagonal Hessian standard error"
        );
    }
}

template<pg::SensitivityOrders Orders, typename Plan, typename Launcher>
Results execute_standalone(const Plan& plan, Launcher&& launcher) {
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    const std::size_t rows = plan.result_count;
    const std::size_t sensitivities = plan.sensitivity_count();

    DeviceArray<typename Plan::Model> models(plan.models);
    DeviceArray<typename Plan::Product> products(plan.products);
    DeviceArray<typename Plan::SensitivitySpec> specs(plan.sensitivities);
    DeviceArray<pg::SensitivityStencil<node_capacity>> stencils(
        rows * sensitivities
    );
    DeviceArray<prep_error::Error> error(1U);
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

    const typename Plan::DeviceInputs inputs{
        models.data,
        models.count,
        products.data,
        products.count,
        specs.data,
        specs.count,
    };
    const monte_carlo::price_gradients::DevicePreparedStencilOutputs<
        node_capacity
    > stencil_outputs{stencils.data, stencils.count, error.data};
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
    const auto launched = launcher(
        plan,
        inputs,
        stencil_outputs,
        launch_configuration(rows),
        outputs
    );
    longstaff_schwartz::validate_regression_diagnostics(
        launched, "standalone Bermudan sensitivity test"
    );
    require(
        error.read()[0U].code == prep_error::valid,
        "Standalone Bermudan sensitivity preparation failed."
    );
    Results result{
        prices.read(),
        price_errors.read(),
        gradients.read(),
        gradient_errors.read(),
        hessians.read(),
        hessian_errors.read(),
    };
    require_valid<Orders>(result, "standalone Bermudan sensitivity");
    return result;
}

template<pg::SensitivityOrders Orders, typename Plan, typename Launcher>
Results execute_curve(const Plan& plan, Launcher&& launcher) {
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    const std::size_t rows = plan.result_count;
    const std::size_t sensitivities = plan.sensitivity_count();

    DeviceArray<typename Plan::Model> models(plan.models);
    DeviceArray<typename Plan::Curve> curves(plan.curves);
    DeviceArray<typename Plan::Product> products(plan.products);
    DeviceArray<typename Plan::SensitivitySpec> specs(plan.sensitivities);
    DeviceArray<pg::SensitivityStencil<node_capacity>> stencils(
        rows * sensitivities
    );
    DeviceArray<prep_error::Error> error(1U);
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

    const typename Plan::DeviceInputs inputs{
        models.data,
        models.count,
        curves.data,
        curves.count,
        products.data,
        products.count,
        specs.data,
        specs.count,
    };
    const monte_carlo::price_gradients::DevicePreparedStencilOutputs<
        node_capacity
    > stencil_outputs{stencils.data, stencils.count, error.data};
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
    const auto launched = launcher(
        plan,
        inputs,
        stencil_outputs,
        launch_configuration(rows),
        outputs
    );
    longstaff_schwartz::validate_regression_diagnostics(
        launched, "curve-fitted Bermudan sensitivity test"
    );
    require(
        error.read()[0U].code == prep_error::valid,
        "Curve-fitted Bermudan sensitivity preparation failed."
    );
    Results result{
        prices.read(),
        price_errors.read(),
        gradients.read(),
        gradient_errors.read(),
        hessians.read(),
        hessian_errors.read(),
    };
    require_valid<Orders>(result, "curve-fitted Bermudan sensitivity");
    return result;
}

template<typename Models, typename Launcher>
Results execute_central_standalone(
    const Models& models,
    Launcher&& launcher
) {
    DeviceArray<typename Models::value_type> device_models(models);
    DeviceArray<product::BermudanSwaptionParameters> device_products(
        kProducts
    );
    DeviceArray<float> prices(models.size());
    DeviceArray<float> errors(models.size());
    const auto launched = launcher(
        device_models.data,
        device_products.data,
        prices.data,
        errors.data
    );
    longstaff_schwartz::validate_regression_diagnostics(
        launched, "Bermudan price-only reference"
    );
    return {prices.read(), errors.read(), {}, {}, {}, {}};
}

template<typename Models, typename Curves, typename Launcher>
Results execute_central_curve(
    const Models& models,
    const Curves& curves,
    Launcher&& launcher
) {
    DeviceArray<typename Models::value_type> device_models(models);
    DeviceArray<typename Curves::value_type> device_curves(curves);
    DeviceArray<product::BermudanSwaptionParameters> device_products(
        kProducts
    );
    DeviceArray<float> prices(models.size());
    DeviceArray<float> errors(models.size());
    const auto launched = launcher(
        device_models.data,
        device_curves.data,
        device_products.data,
        prices.data,
        errors.data
    );
    longstaff_schwartz::validate_regression_diagnostics(
        launched, "curve-fitted Bermudan price-only reference"
    );
    return {prices.read(), errors.read(), {}, {}, {}, {}};
}

void require_same_central(
    const Results& first,
    const Results& second,
    const char* message
) {
    require(
        first.prices.size() == second.prices.size()
            && first.price_errors.size() == second.price_errors.size(),
        "Bermudan central result sizes differ."
    );
    for (std::size_t row = 0U; row < first.prices.size(); ++row) {
        same(first.prices[row], second.prices[row], message);
        same(first.price_errors[row], second.price_errors[row], message);
    }
}

void require_same_first_order(
    const Results& first,
    const Results& diagonal,
    const char* message
) {
    require_same_central(first, diagonal, message);
    require(
        first.gradients.size() == diagonal.gradients.size()
            && first.gradient_errors.size()
                == diagonal.gradient_errors.size(),
        "Bermudan first-order result sizes differ."
    );
    for (std::size_t i = 0U; i < first.gradients.size(); ++i) {
        same(first.gradients[i], diagonal.gradients[i], message);
        same(first.gradient_errors[i], diagonal.gradient_errors[i], message);
    }
}

template<typename Execute, typename Prepare>
void require_reordered_selection(
    const pg::PriceGradientConfiguration& full,
    const Results& reference,
    Execute&& execute,
    Prepare&& prepare
) {
    const pg::PriceGradientConfiguration reordered{{
        full.sensitivities[2U],
        full.sensitivities[0U],
    }};
    const Results selected = execute(prepare(reordered));
    const std::size_t rows = reference.prices.size();
    for (std::size_t row = 0U; row < rows; ++row) {
        same(
            selected.prices[row], reference.prices[row],
            "A reordered sensitivity selection changed the central price."
        );
        same(
            selected.gradients[row * 2U],
            reference.gradients[row * 3U + 2U],
            "A reordered Bermudan sensitivity changed its value."
        );
        same(
            selected.gradients[row * 2U + 1U],
            reference.gradients[row * 3U],
            "A reordered Bermudan sensitivity changed its value."
        );
    }
}

void check_cir() {
    namespace cir = model::fixed_income::cir;
    const std::vector<cir::ModelParameters> models{
        {{0.50f, 0.040f, 0.10f}, 0.040f},
        {{0.30f, 0.025f, 0.08f}, 0.015f},
    };
    const pg::PriceGradientConfiguration full{{
        {"model.mean_reversion", {0.005}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
    }};
    const auto prepare = [&](const auto& selection, pg::SensitivityOrders o) {
        return cir::prepare_cir_bermudan_swaption_sensitivities(
            models, kProducts, PriceConstruction::Aligned,
            kTime, selection, {o}
        );
    };
    const auto execute_first = [&](const auto& plan) {
        return execute_standalone<pg::SensitivityOrders::first>(
            plan,
            [](const auto& p, auto inputs, auto stencils,
               const auto& launch, auto outputs) {
                return cir::launch_cir_bermudan_swaption_price_gradients_cuda<
                    SwaptionSide::payer
                >(p, inputs, stencils, launch, pg::Outputs{
                    outputs.prices,
                    outputs.price_standard_errors,
                    outputs.gradients,
                    outputs.gradient_standard_errors,
                    outputs.price_capacity,
                    outputs.sensitivity_capacity,
                });
            }
        );
    };
    const Results first = execute_first(
        prepare(full, pg::SensitivityOrders::first)
    );
    const Results diagonal = execute_standalone<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(full, pg::SensitivityOrders::first_and_second),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return cir::
                launch_cir_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(p, inputs, stencils, launch, outputs);
        }
    );
    const Results central = execute_central_standalone(
        models,
        [&](auto device_models, auto device_products,
            auto prices, auto errors) {
            return cir::launch_cir_bermudan_swaption_cuda<
                SwaptionSide::payer
            >(
                device_models, models.size(), kProducts.data(),
                device_products, kProducts.size(), PriceConstruction::Aligned,
                models.size(), kPaths,
                kTime.dt * kTime.simulation_steps_per_day,
                kThreads, kBlocks, kSeed, prices, errors
            );
        }
    );
    require_same_central(
        first, central,
        "CIR gradient pipeline changed the central result bits."
    );
    require_same_first_order(
        first, diagonal,
        "CIR diagonal request changed first-order outputs."
    );
    require_reordered_selection(
        full,
        first,
        execute_first,
        [&](const auto& selection) {
            return prepare(selection, pg::SensitivityOrders::first);
        }
    );
}

void check_vasicek() {
    namespace vasicek = model::fixed_income::vasicek;
    const std::vector<vasicek::ModelParameters> models{
        {{0.35f, 0.035f, 0.018f}, 0.030f},
        {{0.70f, 0.025f, 0.012f}, 0.020f},
    };
    const pg::PriceGradientConfiguration full{{
        {"model.long_term_mean", {0.0005, pg::BumpScale::absolute}},
        {"model.volatility", {0.005}},
        {"product.notional", {0.005}},
    }};
    const auto prepare = [&](pg::SensitivityOrders orders) {
        return vasicek::prepare_vasicek_bermudan_swaption_sensitivities(
            models, kProducts, PriceConstruction::Aligned,
            kTime, full, {orders}
        );
    };
    const Results first = execute_standalone<pg::SensitivityOrders::first>(
        prepare(pg::SensitivityOrders::first),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return vasicek::
                launch_vasicek_bermudan_swaption_price_gradients_cuda<
                    SwaptionSide::payer
                >(p, inputs, stencils, launch, pg::Outputs{
                    outputs.prices,
                    outputs.price_standard_errors,
                    outputs.gradients,
                    outputs.gradient_standard_errors,
                    outputs.price_capacity,
                    outputs.sensitivity_capacity,
                });
        }
    );
    const Results diagonal = execute_standalone<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(pg::SensitivityOrders::first_and_second),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return vasicek::
                launch_vasicek_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(p, inputs, stencils, launch, outputs);
        }
    );
    const Results central = execute_central_standalone(
        models,
        [&](auto device_models, auto device_products,
            auto prices, auto errors) {
            return vasicek::launch_vasicek_bermudan_swaption_cuda<
                SwaptionSide::payer
            >(
                device_models, models.size(), kProducts.data(),
                device_products, kProducts.size(), PriceConstruction::Aligned,
                models.size(), kPaths,
                kTime.dt * kTime.simulation_steps_per_day,
                kThreads, kBlocks, kSeed, prices, errors
            );
        }
    );
    require_same_central(
        first, central,
        "Vasicek gradient pipeline changed the central result bits."
    );
    require_same_first_order(
        first, diagonal,
        "Vasicek diagonal request changed first-order outputs."
    );
}

void check_g2() {
    namespace g2 = model::fixed_income::g2;
    const std::vector<g2::ModelParameters> models{
        {{0.19f, 0.010f, 0.47f, 0.014f, 0.20f}, {0.030f, 0.010f}},
        {{0.32f, 0.008f, 0.80f, 0.011f, -0.25f}, {0.020f, -0.005f}},
    };
    const pg::PriceGradientConfiguration full{{
        {"model.correlation", {0.005}},
        {"model.initial_state_y", {0.0005, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005}},
    }};
    const auto prepare = [&](pg::SensitivityOrders orders) {
        return g2::prepare_g2_bermudan_swaption_sensitivities(
            models, kProducts, PriceConstruction::Aligned,
            kTime, full, {orders}
        );
    };
    const Results first = execute_standalone<pg::SensitivityOrders::first>(
        prepare(pg::SensitivityOrders::first),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return g2::launch_g2_bermudan_swaption_price_gradients_cuda<
                SwaptionSide::payer
            >(p, inputs, stencils, launch, pg::Outputs{
                outputs.prices,
                outputs.price_standard_errors,
                outputs.gradients,
                outputs.gradient_standard_errors,
                outputs.price_capacity,
                outputs.sensitivity_capacity,
            });
        }
    );
    const Results diagonal = execute_standalone<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(pg::SensitivityOrders::first_and_second),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return g2::
                launch_g2_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(p, inputs, stencils, launch, outputs);
        }
    );
    const Results node_graph = execute_standalone<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(pg::SensitivityOrders::first_and_second),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            const auto bytes =
                g2::g2_bermudan_swaption_node_graph_workspace_bytes<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(p, launch);
            DeviceArray<std::uint8_t> workspace(bytes);
            return g2::
                launch_g2_bermudan_swaption_node_graph_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(
                    p,
                    inputs,
                    stencils,
                    launch,
                    outputs,
                    workspace.data,
                    workspace.count
                );
        }
    );
    require_same_first_order(
        diagonal,
        node_graph,
        "G2 Bermudan mono and node_graph first-order outputs differ."
    );
    require(
        diagonal.hessians == node_graph.hessians
            && diagonal.hessian_errors == node_graph.hessian_errors,
        "G2 Bermudan mono and node_graph diagonal Hessians differ."
    );

    const auto mixed_plan = g2::prepare_g2_bermudan_swaption_sensitivities(
        models,
        kProducts,
        PriceConstruction::Aligned,
        kTime,
        full,
        pg::SensitivityRequest::full_hessian()
    );
    const auto mixed =
        price_gradient_test::require_mixed_node_graph_parity(
        mixed_plan,
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return g2::
                launch_g2_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(p, inputs, stencils, launch, outputs);
        },
        [](const auto& p, const auto& launch) {
            return g2::
                g2_bermudan_swaption_mixed_node_graph_workspace_bytes<
                    SwaptionSide::payer
                >(p, launch);
        },
        [](const auto& p, auto inputs, auto stencils,
           auto mixed_stencils, const auto& launch,
           auto outputs, auto mixed_outputs,
           void* workspace, std::size_t workspace_bytes) {
            return g2::
                launch_g2_bermudan_swaption_mixed_node_graph_sensitivities_cuda<
                    SwaptionSide::payer
                >(
                    p, inputs, stencils, mixed_stencils,
                    launch, outputs, mixed_outputs,
                    workspace, workspace_bytes
                );
        },
        257U,
        kSeed,
        "G2 Bermudan mixed node graph"
    );

    const auto sparse_mixed_plan =
        g2::prepare_g2_bermudan_swaption_sensitivities(
            models,
            kProducts,
            PriceConstruction::Aligned,
            kTime,
            full,
            pg::SensitivityRequest::selected({}, {}, {{0U, 2U}})
        );
    const auto sparse_mixed =
        price_gradient_test::execute_mixed_node_graph(
            sparse_mixed_plan,
            {
                pg::PricingMethod::monte_carlo,
                0U,
                sparse_mixed_plan.result_count,
                257U,
                kThreads,
                kBlocks,
                kSeed,
                1U,
            },
            [](const auto& p, const auto& launch) {
                return g2::
                    g2_bermudan_swaption_mixed_node_graph_workspace_bytes<
                        SwaptionSide::payer
                    >(p, launch);
            },
            [](const auto& p, auto inputs, auto stencils,
               auto mixed_stencils, const auto& launch,
               auto outputs, auto mixed_outputs,
               void* workspace, std::size_t workspace_bytes) {
                return g2::
                    launch_g2_bermudan_swaption_mixed_node_graph_sensitivities_cuda<
                        SwaptionSide::payer
                    >(
                        p, inputs, stencils, mixed_stencils,
                        launch, outputs, mixed_outputs,
                        workspace, workspace_bytes
                    );
            },
            "G2 Bermudan sparse mixed-only node graph"
        );
    require(
        sparse_mixed_plan.sensitivity_graph.node_capacity == 9U,
        "Mixed-only Bermudan graph retained unselected coordinates."
    );
    price_gradient_test::require_same_bits(
        mixed.prices,
        sparse_mixed.prices,
        "Bermudan sparse mixed-only central price"
    );
    price_gradient_test::require_same_bits(
        mixed.price_errors,
        sparse_mixed.price_errors,
        "Bermudan sparse mixed-only central error"
    );
    for (std::size_t row = 0U; row < models.size(); ++row) {
        price_gradient_test::require_same_bits(
            std::vector<float>{mixed.mixed_hessians[row * 3U + 1U]},
            std::vector<float>{sparse_mixed.mixed_hessians[row]},
            "Bermudan sparse selected mixed Hessian"
        );
    }

    const Results central = execute_central_standalone(
        models,
        [&](auto device_models, auto device_products,
            auto prices, auto errors) {
            return g2::launch_g2_bermudan_swaption_cuda<
                SwaptionSide::payer
            >(
                device_models, models.size(), kProducts.data(),
                device_products, kProducts.size(), PriceConstruction::Aligned,
                models.size(), kPaths,
                kTime.dt * kTime.simulation_steps_per_day,
                kThreads, kBlocks, kSeed, prices, errors
            );
        }
    );
    require_same_central(
        first, central,
        "G2 gradient pipeline changed the central result bits."
    );
    require_same_first_order(
        first, diagonal,
        "G2 diagonal request changed first-order outputs."
    );
}

void check_cir_plus_plus_flat() {
    namespace fitted = model::fixed_income::cir_plus_plus::flat;
    using Model = model::fixed_income::cir_plus_plus::ModelParameters;
    using Curve = curve::flat::FlatCurveParameters;
    const std::vector<Model> models{
        {{0.50f, 0.040f, 0.10f}, 0.040f},
        {{0.30f, 0.025f, 0.08f}, 0.015f},
    };
    const std::vector<Curve> curves{{0.030f}, {-0.005f}};
    const pg::PriceGradientConfiguration selection{{
        {"model.volatility", {0.005}},
        {"curve.rate", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
    }};
    const auto plan =
        fitted::prepare_cir_plus_plus_flat_bermudan_swaption_sensitivities(
            models,
            curves,
            kProducts,
            PriceConstruction::Aligned,
            kTime,
            selection,
            {pg::SensitivityOrders::first_and_second}
        );
    const Results diagonal = execute_curve<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return fitted::
                launch_cir_plus_plus_flat_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(host, inputs, stencils, launch, outputs);
        }
    );
    const Results node_graph = execute_curve<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            const auto bytes = fitted::
                cir_plus_plus_flat_bermudan_swaption_node_graph_workspace_bytes<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(host, launch);
            DeviceArray<std::uint8_t> workspace(bytes);
            return fitted::
                launch_cir_plus_plus_flat_bermudan_swaption_node_graph_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(
                    host,
                    inputs,
                    stencils,
                    launch,
                    outputs,
                    workspace.data,
                    workspace.count
                );
        }
    );
    require_same_first_order(
        diagonal,
        node_graph,
        "CIR++/Flat Bermudan mono and node_graph first-order outputs differ."
    );
    require(
        diagonal.hessians == node_graph.hessians
            && diagonal.hessian_errors == node_graph.hessian_errors,
        "CIR++/Flat Bermudan mono and node_graph diagonal Hessians differ."
    );

    const Results central = execute_central_curve(
        models,
        curves,
        [&](auto device_models, auto device_curves, auto device_products,
            auto prices, auto errors) {
            return fitted::
                launch_cir_plus_plus_flat_bermudan_swaption_cuda<
                    SwaptionSide::payer
                >(
                    device_models,
                    models.size(),
                    device_curves,
                    curves.size(),
                    kProducts.data(),
                    device_products,
                    kProducts.size(),
                    PriceConstruction::Aligned,
                    models.size(),
                    kPaths,
                    kTime.dt * kTime.simulation_steps_per_day,
                    kThreads,
                    kBlocks,
                    kSeed,
                    prices,
                    errors
                );
        }
    );
    require_same_central(
        diagonal,
        central,
        "CIR++/Flat gradient pipeline changed central bits."
    );
}

void check_cir_plus_plus_nelson_siegel() {
    namespace fitted =
        model::fixed_income::cir_plus_plus::nelson_siegel;
    using Model = model::fixed_income::cir_plus_plus::ModelParameters;
    using Curve = curve::nelson_siegel::NelsonSiegelParameters;
    const std::vector<Model> models{
        {{0.50f, 0.040f, 0.10f}, 0.040f},
        {{0.30f, 0.025f, 0.08f}, 0.015f},
    };
    const std::vector<Curve> curves{
        {0.030f, -0.010f, 0.012f, 1.50f},
        {0.025f, -0.008f, 0.010f, 2.00f},
    };
    const pg::PriceGradientConfiguration full{{
        {"model.volatility", {0.005}},
        {"curve.beta1", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
    }};
    const auto prepare = [&](pg::SensitivityOrders orders) {
        return fitted::
            prepare_cir_plus_plus_nelson_siegel_bermudan_swaption_sensitivities(
                models, curves, kProducts, PriceConstruction::Aligned,
                kTime, full, {orders}
            );
    };
    const Results first = execute_curve<pg::SensitivityOrders::first>(
        prepare(pg::SensitivityOrders::first),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return fitted::
                launch_cir_plus_plus_nelson_siegel_bermudan_swaption_price_gradients_cuda<
                    SwaptionSide::payer
                >(p, inputs, stencils, launch, pg::Outputs{
                    outputs.prices,
                    outputs.price_standard_errors,
                    outputs.gradients,
                    outputs.gradient_standard_errors,
                    outputs.price_capacity,
                    outputs.sensitivity_capacity,
                });
        }
    );
    const Results diagonal = execute_curve<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(pg::SensitivityOrders::first_and_second),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return fitted::
                launch_cir_plus_plus_nelson_siegel_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(p, inputs, stencils, launch, outputs);
        }
    );
    const Results node_graph = execute_curve<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(pg::SensitivityOrders::first_and_second),
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            const auto bytes = fitted::
                cir_plus_plus_nelson_siegel_bermudan_swaption_node_graph_workspace_bytes<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(p, launch);
            DeviceArray<std::uint8_t> workspace(bytes);
            return fitted::
                launch_cir_plus_plus_nelson_siegel_bermudan_swaption_node_graph_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(
                    p, inputs, stencils, launch, outputs,
                    workspace.data, workspace.count
                );
        }
    );
    require_same_first_order(
        diagonal,
        node_graph,
        "CIR++/Nelson-Siegel Bermudan mono and node_graph first-order outputs differ."
    );
    require(
        diagonal.hessians == node_graph.hessians
            && diagonal.hessian_errors == node_graph.hessian_errors,
        "CIR++/Nelson-Siegel Bermudan mono and node_graph diagonal Hessians differ."
    );

    const Results central = execute_central_curve(
        models,
        curves,
        [&](auto device_models, auto device_curves, auto device_products,
            auto prices, auto errors) {
            return fitted::
                launch_cir_plus_plus_nelson_siegel_bermudan_swaption_cuda<
                    SwaptionSide::payer
                >(
                    device_models, models.size(), device_curves, curves.size(),
                    kProducts.data(), device_products, kProducts.size(),
                    PriceConstruction::Aligned, models.size(), kPaths,
                    kTime.dt * kTime.simulation_steps_per_day,
                    kThreads, kBlocks, kSeed, prices, errors
                );
        }
    );
    require_same_central(
        first, central,
        "CIR++/Nelson-Siegel gradient pipeline changed central bits."
    );
    require_same_first_order(
        first, diagonal,
        "CIR++/Nelson-Siegel diagonal request changed first-order outputs."
    );
}


void check_g2_plus_plus_svensson_full_mixed_capacity() {
    namespace fitted = model::fixed_income::g2_plus_plus::svensson;
    using Model = model::fixed_income::g2_plus_plus::ModelParameters;
    const std::vector<Model> models{{
        {0.20f, 0.005f, 0.75f, 0.012f, -0.40f}
    }};
    const std::vector<curve::svensson::SvenssonParameters> curves{{
        0.030f, -0.010f, 0.010f, 0.005f, 2.0f, 5.0f
    }};
    const std::vector<product::BermudanSwaptionParameters> products{
        {1.0f, 0.030f, 0.5f, 126U, 126U, 6U, 4U}
    };
    const pg::PriceGradientConfiguration sensitivities{{
        {"model.mean_reversion_x", {0.005f}},
        {"model.volatility_x", {0.0005f, pg::BumpScale::absolute}},
        {"model.mean_reversion_y", {0.005f}},
        {"model.volatility_y", {0.0005f, pg::BumpScale::absolute}},
        {"model.correlation", {0.005f, pg::BumpScale::absolute}},
        {"curve.beta0", {0.0005f, pg::BumpScale::absolute}},
        {"curve.beta1", {0.0005f, pg::BumpScale::absolute}},
        {"curve.beta2", {0.0005f, pg::BumpScale::absolute}},
        {"curve.beta3", {0.0005f, pg::BumpScale::absolute}},
        {"curve.tau1", {0.005f}},
        {"curve.tau2", {0.005f}},
        {"product.notional", {0.005f}},
        {"product.strike", {0.0005f, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005f}},
    }};
    const auto plan =
        fitted::prepare_g2_plus_plus_svensson_bermudan_swaption_sensitivities(
            models,
            curves,
            products,
            PriceConstruction::Aligned,
            kTime,
            sensitivities,
            pg::SensitivityRequest::full_hessian()
        );
    require(
        plan.sensitivity_graph.mixed_second.size() == 91U
            && plan.sensitivity_graph.node_capacity == 407U,
        "G2++/Svensson full Hessian graph capacity is incorrect."
    );
    price_gradient_test::require_mixed_node_graph_parity(
        plan,
        [](const auto& p, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return fitted::
                launch_g2_plus_plus_svensson_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(p, inputs, stencils, launch, outputs);
        },
        [](const auto& p, const auto& launch) {
            return fitted::
                g2_plus_plus_svensson_bermudan_swaption_mixed_node_graph_workspace_bytes<
                    SwaptionSide::payer
                >(p, launch);
        },
        [](const auto& p, auto inputs, auto stencils,
           auto mixed_stencils, const auto& launch,
           auto outputs, auto mixed_outputs,
           void* workspace, std::size_t workspace_bytes) {
            return fitted::
                launch_g2_plus_plus_svensson_bermudan_swaption_mixed_node_graph_sensitivities_cuda<
                    SwaptionSide::payer
                >(
                    p, inputs, stencils, mixed_stencils,
                    launch, outputs, mixed_outputs,
                    workspace, workspace_bytes
                );
        },
        257U,
        kSeed,
        "G2++/Svensson maximum mixed Bermudan graph"
    );
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "Bermudan sensitivity cudaGetDeviceCount");

    check_cir();
    check_vasicek();
    check_g2();
    check_cir_plus_plus_flat();
    check_cir_plus_plus_nelson_siegel();
    check_g2_plus_plus_svensson_full_mixed_capacity();
}
