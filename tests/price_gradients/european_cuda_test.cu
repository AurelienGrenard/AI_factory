// Public selected-gradient checks: historical parity, analytic Greeks, subsets and launch guards.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/product/european_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/european_option_price_delta.cuh"
#include "cuda_test_support.cuh"
#include "diagonal_cuda_test_support.cuh"
#include "closed_form_mixed_cuda_test_support.cuh"
#include <bit>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace bs = model::equity::black_scholes;
namespace hs = model::equity::heston;
std::size_t bs_paths = 1U << 17U;
std::size_t heston_paths = 4097U;
using namespace price_gradient_test;

template<OptionSide Side, pg::SensitivityOrders Orders>
DiagonalResults execute_heston_diagonal(
    const hs::EuropeanOptionPriceGradientPlan& plan,
    pg::LaunchConfiguration launch
) {
    return execute_diagonal<Orders>(
        plan,
        launch,
        hs::launch_heston_european_option_diagonal_sensitivities_cuda<
            Side, Orders
        >
    );
}

template<OptionSide Side> void black_scholes() {
    const std::vector<bs::ModelParameters> models{{.75f,.03f,.01f,.2f},{1.2f,0.f,0.f,.3f},{1.4f,.04f,.02f,.25f}};
    const std::vector<product::EuropeanOptionParameters> products{{1.f,252U},{1.2f,126U},{1.1f,63U}};
    const pg::Sensitivity spot{"model.spot", {.005f}};
    const auto prepare = [&](const pg::PriceGradientConfiguration& selected) {
        return bs::prepare_black_scholes_european_option_price_gradients(models, products, PriceConstruction::Aligned, {}, selected);
    };
    auto launcher = bs::launch_black_scholes_european_option_price_gradients_cuda<Side>;
    pg::LaunchConfiguration launch{pg::PricingMethod::closed_form, 0U, 3U, bs_paths, 128U, 1U, 719U};
    const auto solo = execute(prepare({{spot}}), launch, launcher, true);
    DeviceArray<bs::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    DeviceArray<float> legacy(6U);
    bs::launch_black_scholes_european_option_price_delta_cuda<Side>(models.data(), device_models.data, 3U,
        device_products.data, 3U, PriceConstruction::Aligned, 3U, 0U, 3U, 1.f/252.f, 128U, 1U, {.01f}, legacy.data, legacy.data+3U);
    const auto reference = legacy.read();
    for (unsigned int row = 0; row < 3; ++row) {
        same(solo.price[row], reference[row], "Black-Scholes central price parity");
        same(solo.gradient[row], reference[3U+row], "Black-Scholes spot delta parity");
    }
    const pg::PriceGradientConfiguration full{{spot,
        {"model.volatility", {.002f}}, {"product.strike", {.002f}},
        {"model.risk_free_rate", {.0001f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {.0001f, pg::BumpScale::absolute}},
        {"product.maturity_years", {1.f/504.f, pg::BumpScale::absolute}}}};
    const auto analytical = execute(prepare(full), launch, launcher);
    const double sign = Side == OptionSide::call ? 1.0 : -1.0;
    for (unsigned int row = 0; row < 3; ++row) {
        same(analytical.price[row], solo.price[row], "Adding gradients changed closed-form price");
        same(analytical.gradient[row*6U], solo.gradient[row], "Adding gradients changed closed-form delta");
        const auto& m = models[row];
        const double t = products[row].maturity_days/252.0, strike = products[row].strike;
        const double d1 = (std::log(m.spot/strike)+(m.risk_free_rate-m.dividend_yield+.5*m.volatility*m.volatility)*t)/(m.volatility*std::sqrt(t));
        const double d2 = d1-m.volatility*std::sqrt(t);
        const auto cdf = [](double x) { return .5*std::erfc(-x/std::sqrt(2.0)); };
        const double dq = std::exp(-m.dividend_yield*t), dr = std::exp(-m.risk_free_rate*t);
        const double density = std::exp(-.5*d1*d1)/std::sqrt(2.0*std::acos(-1.0));
        const double expected[6]{sign*dq*cdf(sign*d1), m.spot*dq*density*std::sqrt(t),
            -sign*dr*cdf(sign*d2), sign*strike*t*dr*cdf(sign*d2), -sign*m.spot*t*dq*cdf(sign*d1),
            m.spot*dq*density*m.volatility/(2*std::sqrt(t))
                + sign*m.risk_free_rate*strike*dr*cdf(sign*d2)-sign*m.dividend_yield*m.spot*dq*cdf(sign*d1)};
        for (unsigned int i = 0; i < 6; ++i)
            require(std::abs(analytical.gradient[row*6U+i]-expected[i]) < 1.5e-3, "Analytical gradient mismatch.");
    }
    launch.method = pg::PricingMethod::monte_carlo;
    const auto mc = execute(prepare(full), launch, launcher, true);
    selected_prefixes(full,mc,launch,prepare,launcher);
    const auto mc_solo = execute(prepare({{spot}}), launch, launcher);
    for (unsigned int row = 0; row < 3; ++row) {
        same(mc.price[row], mc_solo.price[row], "Adding gradients changed MC price");
        same(mc.price_error[row], mc_solo.price_error[row], "Adding gradients changed MC price error");
        same(mc.gradient[row*6U], mc_solo.gradient[row], "Adding gradients changed MC delta");
        same(mc.gradient_error[row*6U], mc_solo.gradient_error[row], "Adding gradients changed MC delta error");
        for (unsigned int i = 0; i < 6; ++i)
            require(std::abs(mc.gradient[row*6U+i]-analytical.gradient[row*6U+i])
                < 6*mc.gradient_error[row*6U+i]+.002, "MC gradient differs from analytical value beyond paired noise budget.");
    }
    for (unsigned int i = 1; i < 6; ++i) {
        const auto single = execute(prepare({{full.sensitivities[i]}}),launch,launcher);
        for (unsigned int row = 0; row < 3; ++row) {
            same(mc.gradient[row*6U+i],single.gradient[row],full.sensitivities[i].parameter.c_str());
            same(mc.gradient_error[row*6U+i],single.gradient_error[row],full.sensitivities[i].parameter.c_str());
        }
    }
    const auto reversed = execute(prepare({{{"product.strike", {.002f}}, spot}}), launch, launcher);
    for (unsigned int row = 0; row < 3; ++row) {
        same(reversed.gradient[row*2U+1U], mc_solo.gradient[row], "Selection order changed spot delta");
        same(reversed.gradient[row*2U], mc.gradient[row*6U+2U], "Selection order changed strike gradient");
    }
    const auto empty = execute(prepare({}), launch, launcher);
    for (unsigned int row = 0; row < 3; ++row) same(empty.price[row], mc_solo.price[row], "Empty selection changed price");
    std::cout << "Black-Scholes " << option_side_name(Side) << ": CF parity, six analytical/MC gradients and selection invariance passed\n";
}

template<OptionSide Side> void black_scholes_diagonal() {
    const std::vector<bs::ModelParameters> models{
        {.75f,.03f,.01f,.2f}, {1.2f,0.f,0.f,.3f}, {1.4f,.04f,.02f,.25f}
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.f,252U}, {1.2f,126U}, {1.1f,63U}
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f}},
        {"model.volatility", {.002f}},
        {"product.strike", {.002f}},
    }};
    const auto prepare = [&](pg::SensitivityOrders orders) {
        return bs::prepare_black_scholes_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, selection,
            {orders}
        );
    };
    pg::LaunchConfiguration launch{
        pg::PricingMethod::closed_form, 0U, models.size(), 0U,
        128U, 1U, 0U, 1U
    };
    const auto combined = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(pg::SensitivityOrders::first_and_second), launch,
        bs::launch_black_scholes_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto second = execute_diagonal<pg::SensitivityOrders::second>(
        prepare(pg::SensitivityOrders::second), launch,
        bs::launch_black_scholes_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::second
        >
    );
    const auto mixed_plan =
        bs::prepare_black_scholes_european_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            {},
            selection,
            pg::SensitivityRequest::full_hessian()
        );
    const auto mixed = execute_closed_form_mixed(
        mixed_plan,
        launch,
        [](const auto& plan, const auto& configuration) {
            return bs::
                black_scholes_european_option_mixed_node_graph_workspace_bytes<
                    Side
                >(plan, configuration);
        },
        [](const auto& plan, auto inputs, auto stencils,
           auto mixed_stencils, const auto& configuration,
           auto outputs, auto mixed_outputs,
           void* workspace, std::size_t workspace_bytes) {
            bs::
                launch_black_scholes_european_option_mixed_node_graph_sensitivities_cuda<
                    Side
                >(
                    plan,
                    inputs,
                    stencils,
                    mixed_stencils,
                    configuration,
                    outputs,
                    mixed_outputs,
                    workspace,
                    workspace_bytes
                );
        },
        "Black-Scholes closed-form mixed Hessian"
    );
    require_closed_form_diagonal_parity(
        combined, mixed, "Black-Scholes closed-form mixed/diagonal"
    );

    const auto sparse_plan =
        bs::prepare_black_scholes_european_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            {},
            selection,
            pg::SensitivityRequest::selected({}, {}, {{0U, 2U}})
        );
    const auto sparse = execute_closed_form_mixed(
        sparse_plan,
        launch,
        [](const auto& plan, const auto& configuration) {
            return bs::
                black_scholes_european_option_mixed_node_graph_workspace_bytes<
                    Side
                >(plan, configuration);
        },
        [](const auto& plan, auto inputs, auto stencils,
           auto mixed_stencils, const auto& configuration,
           auto outputs, auto mixed_outputs,
           void* workspace, std::size_t workspace_bytes) {
            bs::
                launch_black_scholes_european_option_mixed_node_graph_sensitivities_cuda<
                    Side
                >(
                    plan,
                    inputs,
                    stencils,
                    mixed_stencils,
                    configuration,
                    outputs,
                    mixed_outputs,
                    workspace,
                    workspace_bytes
                );
        },
        "Black-Scholes sparse mixed Hessian"
    );
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(
            sparse.mixed_hessians[row],
            mixed.mixed_hessians[row * 3U + 1U],
            "Black-Scholes sparse mixed selection changed its value"
        );
    }
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(combined.price[row], second.price[row],
             "Black-Scholes sensitivity order changed price");
        const double maturity = products[row].maturity_days / 252.0;
        const auto& model = models[row];
        const double d1 = (
            std::log(model.spot/products[row].strike)
            + (model.risk_free_rate-model.dividend_yield
               + .5*model.volatility*model.volatility)*maturity
        ) / (model.volatility*std::sqrt(maturity));
        const double density = std::exp(-.5*d1*d1)
            / std::sqrt(2.0*std::acos(-1.0));
        const double gamma = std::exp(-model.dividend_yield*maturity)*density
            / (model.spot*model.volatility*std::sqrt(maturity));
        if (std::abs(combined.diagonal_hessian[row*3U]-gamma) >= 5e-3) {
            std::cerr << "Black-Scholes represented gamma row=" << row
                      << " estimate=" << combined.diagonal_hessian[row*3U]
                      << " analytic=" << gamma << '\n';
            throw std::runtime_error(
                "Black-Scholes represented spot gamma mismatch."
            );
        }
        for (std::size_t i = 0U; i < 3U; ++i) {
            const auto index = row*3U+i;
            same(combined.diagonal_hessian[index],
                 second.diagonal_hessian[index],
                 "Black-Scholes order changed diagonal Hessian");
            require(std::isfinite(combined.gradient[index])
                    && std::isfinite(combined.diagonal_hessian[index]),
                    "Black-Scholes diagonal sensitivity is non-finite.");
        }
    }
}

template<OptionSide Side> void heston() {
    const std::vector<hs::ModelParameters> models{{.75f,.03f,.01f,.04f,1.5f,.04f,.3f,-.7f},{1.2f,0.f,0.f,.06f,.8f,.04f,.4f,-.3f},{1.4f,.01f,0.f,0.f,1.f,.04f,.3f,1.f}};
    const std::vector<product::EuropeanOptionParameters> products{{.8f,16U},{1.2f,12U},{1.1f,8U}};
    auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return hs::prepare_heston_european_option_price_gradients(models, products, PriceConstruction::Aligned, {}, selection);
    };
    const pg::Sensitivity spot{"model.spot", {.005f}};
    auto launcher = hs::launch_heston_european_option_price_gradients_cuda<Side>;
    DeviceArray<hs::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    for (unsigned int threads : {64U, 128U, 256U}) {
        pg::LaunchConfiguration launch{pg::PricingMethod::monte_carlo,0U,3U,heston_paths,threads,1U,713U};
        const auto solo = execute(prepare({{spot}}), launch, launcher, true);
        // Independent AnalyticHestonEngine references on the exact 252-day
        // maturity grid (QuantLib 1.43, Business252/NullCalendar). These check
        // the price level; the legacy launcher below only checks code parity.
        if (threads == 128U && heston_paths >= 4097U) {
            constexpr double call_reference[2]{
                0.0013678667281647415, 0.025358097171288174
            };
            constexpr double put_reference[2]{
                0.050321546867132215, 0.025358097171288174
            };
            const auto* independent_reference = Side == OptionSide::call
                ? call_reference : put_reference;
            for (unsigned int row = 0U; row < 2U; ++row) {
                const double tolerance = 8.0 * solo.price_error[row] + 0.0005;
                if (std::abs(solo.price[row] - independent_reference[row])
                    >= tolerance) {
                    std::cerr << "Heston analytic price mismatch row=" << row
                              << " observed=" << solo.price[row]
                              << " reference=" << independent_reference[row]
                              << " standard_error=" << solo.price_error[row]
                              << '\n';
                    throw std::runtime_error(
                        "Heston Monte Carlo price misses analytic reference."
                    );
                }
            }
        }
        DeviceArray<float> old(12U);
        hs::launch_heston_european_option_price_delta_cuda<Side>(models.data(),device_models.data,3U,
            products.data(),device_products.data,3U,PriceConstruction::Aligned,3U,0U,3U,heston_paths,1.f/504.f,2U,
            threads,1U,713U,{.01f},old.data,old.data+3U,old.data+6U,old.data+9U);
        const auto reference = old.read();
        for (unsigned int row = 0; row < 3; ++row) {
            same(solo.price[row],reference[row],"Heston price parity");
            same(solo.price_error[row],reference[3U+row],"Heston price error parity");
            same(solo.gradient[row],reference[6U+row],"Heston delta parity");
            same(solo.gradient_error[row],reference[9U+row],"Heston delta error parity");
        }
        const auto extended = execute(prepare({{{"model.initial_variance", {.001f,pg::BumpScale::absolute}},
            {"model.rho", {.002f,pg::BumpScale::absolute}},spot,{"product.strike",{.002f}}}}),launch,launcher);
        for (unsigned int row = 0; row < 3; ++row) {
            same(extended.price[row],solo.price[row],"Heston subset price parity");
            same(extended.gradient[row*4U+2U],solo.gradient[row],"Heston subset delta parity");
            same(extended.gradient_error[row*4U+2U],solo.gradient_error[row],"Heston subset delta error parity");
        }
        const pg::PriceGradientConfiguration full{{spot,
            {"model.risk_free_rate", {.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield", {.0001,pg::BumpScale::absolute}},
            {"model.initial_variance", {.001,pg::BumpScale::absolute}},
            {"model.kappa", {.005}}, {"model.theta", {.005}}, {"model.gamma", {.005}},
            {"model.rho", {.002,pg::BumpScale::absolute}}, {"product.strike", {.002}},
            {"product.maturity_years", {1.f/504.f,pg::BumpScale::absolute}}}};
        const auto all = execute(prepare(full),launch,launcher);
        const auto empty = execute(prepare({}),launch,launcher);
        for (unsigned int row = 0; row < 3; ++row) {
            same(all.price[row],solo.price[row],"Heston full-selection price parity");
            same(empty.price[row],solo.price[row],"Heston empty-selection price parity");
            same(all.gradient[row*10U],solo.gradient[row],"Heston full-selection delta parity");
            same(all.gradient_error[row*10U],solo.gradient_error[row],"Heston full-selection delta error parity");
        }
        selected_prefixes(full,all,launch,prepare,launcher);
        if (threads == 128U) {
            for (unsigned int i = 1; i < 10; ++i) {
                const auto single = execute(prepare({{full.sensitivities[i]}}),launch,launcher);
                for (unsigned int row = 0; row < 3; ++row) {
                    same(all.gradient[row*10U+i],single.gradient[row],full.sensitivities[i].parameter.c_str());
                    same(all.gradient_error[row*10U+i],single.gradient_error[row],full.sensitivities[i].parameter.c_str());
                }
            }
        }
    }
    std::cout << "Heston " << option_side_name(Side) << ": historical four-output parity, ten coordinates, reordered subsets, boundaries and grid maturities passed\n";
}

template<OptionSide Side> void heston_diagonal() {
    const std::vector<hs::ModelParameters> models{
        {.75f,.0f,.0f,.0f,1.5f,.04f,.3f,-.7f},
        {1.2f,.0f,.0f,.06f,.8f,.04f,.4f,-.3f},
        {1.4f,.0f,.0f,.04f,1.f,.04f,.3f,1.f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {.8f,16U}, {1.2f,12U}, {1.1f,8U}
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f}},
        {"model.initial_variance", {.001f, pg::BumpScale::absolute}},
        {"model.kappa", {.005f}},
        {"model.theta", {.005f}},
        {"model.gamma", {.005f}},
        {"model.rho", {.002f, pg::BumpScale::absolute}},
        {"product.strike", {.002f}},
    }};
    const auto prepare = [&](
        const pg::PriceGradientConfiguration& selected,
        pg::SensitivityOrders orders
    ) {
        return hs::prepare_heston_european_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            {},
            selected,
            {orders}
        );
    };
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        models.size(),
        heston_paths,
        128U,
        1U,
        1709U,
        1U,
    };
    const auto combined = execute_heston_diagonal<
        Side,
        pg::SensitivityOrders::first_and_second
    >(prepare(selection, pg::SensitivityOrders::first_and_second), launch);
    const auto second_only = execute_heston_diagonal<
        Side,
        pg::SensitivityOrders::second
    >(prepare(selection, pg::SensitivityOrders::second), launch);
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(
            combined.price[row],
            second_only.price[row],
            "Sensitivity order changed Heston price"
        );
        same(
            combined.price_error[row],
            second_only.price_error[row],
            "Sensitivity order changed Heston price error"
        );
        for (std::size_t sensitivity = 0U;
             sensitivity < selection.sensitivities.size();
             ++sensitivity) {
            const auto index = row*selection.sensitivities.size()+sensitivity;
            same(
                combined.diagonal_hessian[index],
                second_only.diagonal_hessian[index],
                "Second-order request changed diagonal Hessian"
            );
            same(
                combined.diagonal_hessian_error[index],
                second_only.diagonal_hessian_error[index],
                "Second-order request changed diagonal-Hessian error"
            );
            require(
                std::isfinite(combined.gradient[index]),
                "Non-finite Heston gradient."
            );
            require(
                std::isfinite(combined.diagonal_hessian[index]),
                "Non-finite Heston diagonal Hessian."
            );
        }
    }
    for (std::size_t sensitivity = 0U;
         sensitivity < selection.sensitivities.size();
         ++sensitivity) {
        const pg::PriceGradientConfiguration single{{
            selection.sensitivities[sensitivity]
        }};
        const auto result = execute_heston_diagonal<
            Side,
            pg::SensitivityOrders::first_and_second
        >(
            prepare(single, pg::SensitivityOrders::first_and_second), launch
        );
        for (std::size_t row = 0U; row < models.size(); ++row) {
            const auto index = row*selection.sensitivities.size()+sensitivity;
            same(
                combined.gradient[index],
                result.gradient[row],
                "Selection changed Heston gradient"
            );
            same(
                combined.gradient_error[index],
                result.gradient_error[row],
                "Selection changed Heston gradient error"
            );
            same(
                combined.diagonal_hessian[index],
                result.diagonal_hessian[row],
                "Selection changed Heston diagonal Hessian"
            );
            same(
                combined.diagonal_hessian_error[index],
                result.diagonal_hessian_error[row],
                "Selection changed Heston diagonal-Hessian error"
            );
        }
    }
    const pg::PriceGradientConfiguration maturity{{{
        "product.maturity_years",
        {1.f/504.f, pg::BumpScale::absolute}
    }}};
    const auto maturity_result = execute_heston_diagonal<
        Side, pg::SensitivityOrders::first_and_second
    >(
        prepare(maturity, pg::SensitivityOrders::first_and_second), launch
    );
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(maturity_result.price[row], combined.price[row],
             "Heston maturity diagonal changed central price");
        require(std::isfinite(maturity_result.gradient[row])
                && std::isfinite(maturity_result.diagonal_hessian[row]),
                "Heston maturity diagonal is non-finite.");
    }
    std::cout << "Heston " << option_side_name(Side)
              << ": seven gradients and diagonal Hessians, boundaries and selection invariance passed\n";
}

void heston_stencil_preparation() {
    const std::vector<hs::ModelParameters> models{
        {1.f,0.f,0.f,0.f,1.5f,.04f,.3f,-.7f},
        {1.2f,0.f,0.f,.06f,.8f,.04f,.4f,1.f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.f,16U}, {1.1f,12U}
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.initial_variance", {.001f, pg::BumpScale::absolute}},
        {"model.rho", {.002f, pg::BumpScale::absolute}},
        {"product.strike", {.005f}},
    }};
    const auto plan = hs::prepare_heston_european_option_price_gradients(
        models, products, PriceConstruction::Aligned, {}, selection
    );
    DeviceArray<hs::ModelParameters> device_models(plan.models);
    DeviceArray<product::EuropeanOptionParameters> device_products(
        plan.products
    );
    DeviceArray<hs::EuropeanOptionPriceGradientPlan::SensitivitySpec>
        device_sensitivities(plan.sensitivities);
    DeviceArray<pg::SensitivityStencil<3U>> device_stencils(
        plan.result_count*plan.sensitivity_count()
    );
    DeviceArray<equity::price_gradients::device_preparation::Error> error(1U);
    const hs::EuropeanOptionPriceGradientPlan::DeviceInputs inputs{
        device_models.data,
        device_models.count,
        device_products.data,
        device_products.count,
        device_sensitivities.data,
        device_sensitivities.count,
    };
    const hs::EuropeanOptionPriceGradientPlan::StencilOutputs outputs{
        device_stencils.data, device_stencils.count, error.data
    };
    hs::prepare_european_option_price_gradient_stencils_cuda(
        plan, inputs, outputs, 0U, plan.result_count
    );
    check_cuda(cudaDeviceSynchronize(), "Heston stencil-only preparation");
    require(error.read()[0U].code == 0, "Stencil-only preparation failed.");
    const auto actual = device_stencils.read();
    for (std::size_t row = 0U; row < plan.result_count; ++row) {
        hs::EuropeanOptionPriceGradientPlan::Preparation::Scenario central{};
        require(
            hs::EuropeanOptionPriceGradientPlan::Preparation::make_central(
                plan.models[row], plan.products[row], plan.time, central
            ),
            "Host stencil oracle central preparation failed."
        );
        for (std::size_t sensitivity = 0U;
             sensitivity < plan.sensitivity_count();
             ++sensitivity) {
            pg::SensitivityTask<
                hs::EuropeanOptionPriceGradientPlan::Preparation::Scenario,
                3U
            > task{};
            int preparation_error =
                equity::price_gradients::device_preparation::valid;
            require(
                equity::price_gradients::device_preparation::
                    build_sensitivity_task<
                        pg::SensitivityOrders::first,
                        hs::EuropeanOptionPriceGradientPlan::Preparation
                    >(
                        central,
                        plan.sensitivities[sensitivity],
                        plan.time,
                        task,
                        preparation_error
                    ),
                "Host stencil oracle failed."
            );
            const auto& observed =
                actual[row*plan.sensitivity_count()+sensitivity];
            require(
                std::memcmp(&observed, &task.stencil, sizeof(observed)) == 0,
                "Stencil-only device preparation differs from its host mirror."
            );
        }
    }
    std::cout << "Heston checkpoint stencil reconstruction passed\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            // Keep every model/side/batch/tail/geometry case and multiple path
            // iterations, including a partial final iteration at 256 threads.
            bs_paths = 1025U;
            heston_paths = 513U;
            std::cout << "Sanitizer fixture: BS paths=1025, Heston paths=513; complete case matrix\n";
        } else if (argc != 1) throw std::invalid_argument("Usage: test_price_gradients_european_cuda [--sanitizer]");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        black_scholes<OptionSide::call>(); black_scholes<OptionSide::put>();
        black_scholes_diagonal<OptionSide::call>();
        black_scholes_diagonal<OptionSide::put>();
        heston<OptionSide::call>(); heston<OptionSide::put>();
        heston_diagonal<OptionSide::call>();
        heston_diagonal<OptionSide::put>();
        heston_stencil_preparation();
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
