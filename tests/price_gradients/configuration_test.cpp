// Independent polynomial, domain, selection and Brownian covariance checks for host plans.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/product/double_knock_out_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/cev/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"
#include "model/fixed_income/cir/product/european_swaption_price_gradients.cuh"
#include "price_gradients/result.cuh"
#include "price_gradients/reconstruction.cuh"
#include "equity/spot_relative_product.hpp"
#include "product/double_knock_out_option/parameters.hpp"
#include "product/up_one_touch/parameters.hpp"
#include "product/range_accrual/parameters.hpp"
#include "product/athena_autocall/parameters.hpp"
#include "price_gradients/row_mapping.cuh"
#include <iostream>

namespace {
using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace bs = model::equity::black_scholes;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<typename Function> void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Invalid configuration was accepted.");
}
template<typename Plan>
typename Plan::Preparation::Scenario central_row(
    const Plan& plan,
    std::size_t row
) {
    const auto indices = pg::price_row_indices(
        row, plan.construction, plan.products.size()
    );
    typename Plan::Preparation::Scenario central{};
    require(
        Plan::Preparation::make_central(
            plan.models[indices.model],
            equity::spot_relative::selected_central_product(
                plan.models[indices.model], plan.products[indices.product],
                plan.relative_quotes),
            plan.time,
            central
        ),
        "Compact central row could not be reconstructed."
    );
    return central;
}
template<typename Plan>
auto first_order_task(
    const Plan& plan,
    std::size_t row,
    std::size_t sensitivity
) {
    pg::SensitivityTask<typename Plan::Preparation::Scenario, 3U> task{};
    int error = pg::device_preparation::valid;
    const auto central = central_row(plan, row);
    require(
        pg::device_preparation::
            build_sensitivity_task<
                pg::SensitivityOrders::first,
                typename Plan::Preparation
            >(
                central,
                plan.sensitivities[sensitivity],
                plan.time,
                task,
                error
            ),
        "Compact sensitivity task could not be reconstructed."
    );
    return task;
}
void stencils() {
    const pg::BumpConfiguration bump{0.125f, pg::BumpScale::absolute};
    for (float x : {0.0f, 0.0625f, 0.5f, 0.9375f, 1.0f}) {
        const auto stencil = pg::prepare_stencil(x, bump, [](float p) { return p >= 0 && p <= 1; });
        require(std::abs(pg::difference(stencil, x*x, stencil.first*stencil.first, stencil.second*stencil.second) - 2*x) < 2.e-6f,
            "Quadratic derivative does not match analytical value.");
        require(pg::difference(stencil, 3.f, 3.f, 3.f) == 0.f, "Constant difference is not zero.");
        require(stencil.kind == (x < .125f ? pg::StencilKind::forward : x > .875f ? pg::StencilKind::backward : pg::StencilKind::centered),
            "Wrong boundary orientation.");
    }
    rejects([&] { pg::prepare_stencil(0.f, {.01f, pg::BumpScale::relative}, [](float) { return true; }); });
    rejects([&] { pg::prepare_stencil(0.f, {.1f, pg::BumpScale::absolute, pg::BoundaryRule::central_only}, [](float p) { return p >= 0; }); });
    rejects([&] { pg::prepare_stencil(.5f, {1.f, pg::BumpScale::absolute}, [](float p) { return p >= 0 && p <= 1; }); });
    rejects([&] { pg::prepare_stencil(1.f, {1.e-12f, pg::BumpScale::absolute}, [](float) { return true; }); });
    // Nonuniform represented endpoints still reproduce a quadratic derivative.
    const auto uneven = pg::prepare_stencil(1.f, bump, [](float p) { return p >= 1.f; },
        [](int multiple) { return multiple == 1 ? 1.125f : multiple == 2 ? 1.375f : .875f; });
    require(std::abs(pg::difference(uneven, 1.f, uneven.first*uneven.first, uneven.second*uneven.second)-2.f)<2.e-6f,
        "Nonuniform stencil coefficients are incorrect.");
}
void covariance() {
    for (auto dates : {std::array<float, 3>{1.f,.8f,1.2f}, {1.f,1.1f,1.2f}, {1.f,.9f,.8f}}) {
        std::array<std::array<float, 3>, 3> weights{{
            {1.0f, 0.0f, 0.0f}, {}, {}
        }};
        pg::device_preparation::brownian_endpoint_weights(
            dates[0], dates[1], dates[2],
            weights[1].data(), weights[2].data()
        );
        for (unsigned int i = 0; i < 3; ++i) for (unsigned int j = 0; j < 3; ++j) {
            double covariance = 0;
            for (unsigned int n = 0; n < 3; ++n) covariance += weights[i][n]*weights[j][n];
            covariance *= std::sqrt(dates[i]*dates[j]);
            require(std::abs(covariance-std::min(dates[i],dates[j])) < 2.e-7, "Endpoint covariance is not Brownian.");
        }
    }
}
void selection() {
    const std::array<bs::ModelParameters, 2> models{{{1.f,0.f,0.f,.2f},{1.2f,0.f,0.f,.3f}}};
    const std::array<product::EuropeanOptionParameters, 2> products{{{1.f,252U},{.9f,7U}}};
    const pg::PriceGradientConfiguration selected{{{"model.spot", {.005f}}, {"product.strike", {.01f}}}};
    const auto prepare = [&](const auto& config) { return bs::prepare_black_scholes_european_option_price_gradients(models, products,
        PriceConstruction::CartesianProduct, {}, config); };
    const auto plan = prepare(selected);
    require(plan.result_count == 4U && plan.models.size() == 2U
            && plan.products.size() == 2U && plan.sensitivities.size() == 2U
            && plan.construction == PriceConstruction::CartesianProduct
            && plan.relative_quotes,
        "Cartesian quote mapping was not retained compactly.");
    for (std::size_t row = 0U; row < plan.result_count; ++row) {
        const auto central = central_row(plan, row);
        require(
            central.model.risk_free_rate == 0.f
                && central.model.dividend_yield == 0.f,
            "Unselected zero parameters were changed."
        );
    }
    require(central_row(plan, 1U).product.strike == .9f
            && central_row(plan, 2U).model.spot == 1.2f
            && std::abs(central_row(plan, 3U).product.strike - 1.08f)
                < 1.e-6f,
        "Cartesian quote resolution or ordering changed.");
    require(first_order_task(plan, 0U, 0U).central_requirement
                == pg::CentralRequirement::state
        && first_order_task(plan, 0U, 1U).central_requirement
                == pg::CentralRequirement::state,
        "Homogeneous spot/strike should reuse state without central payoff.");
    const auto volatility = prepare(pg::PriceGradientConfiguration{{{"model.volatility", {.01f}}}});
    require(first_order_task(volatility, 0U, 0U).central_requirement
                == pg::CentralRequirement::none,
        "Centered dynamics gradient incorrectly requires central state.");
    const auto boundary = prepare(pg::PriceGradientConfiguration{{{"model.volatility", {.3f, pg::BumpScale::absolute}}}});
    const auto boundary_task = first_order_task(boundary, 0U, 0U);
    require(boundary_task.stencil.kind != pg::StencilKind::centered
        && boundary_task.central_requirement == pg::CentralRequirement::payoff,
        "Boundary stencil does not declare its central payoff dependency.");
    const auto price_only = prepare(pg::PriceGradientConfiguration{});
    require(price_only.result_count == 4U && price_only.sensitivities.empty(),
        "Empty selection does not produce a compact price-only plan.");
    rejects([&] { prepare(pg::PriceGradientConfiguration{{{"model.spot", {.01f}}, {"model.spot", {.02f}}}}); });
    rejects([&] { prepare(pg::PriceGradientConfiguration{{{"model.jump_intensity", {.01f}}}}); });
    const auto dates = prepare(pg::PriceGradientConfiguration{{{"product.maturity_years", {1.f/504.f, pg::BumpScale::absolute}}}});
    const auto date_task = first_order_task(dates, 0U, 0U);
    require(date_task.nodes[1U].step_count == 503U
            && date_task.nodes[2U].step_count == 505U,
        "One-step maturity bump changed calendar units.");
    const auto misaligned_dates = prepare(pg::PriceGradientConfiguration{{{
        "product.maturity_years",
        {.5f/504.f, pg::BumpScale::absolute}
    }}});
    pg::SensitivityTask<
        typename decltype(misaligned_dates)::Preparation::Scenario,
        3U
    > misaligned_task{};
    int misaligned_error = pg::device_preparation::valid;
    require(
        !pg::device_preparation::
            build_sensitivity_task<
                pg::SensitivityOrders::first,
                typename decltype(misaligned_dates)::Preparation
            >(
                central_row(misaligned_dates, 0U),
                misaligned_dates.sensitivities[0U],
                misaligned_dates.time,
                misaligned_task,
                misaligned_error
            ),
        "A sub-calendar maturity bump was accepted by device preparation."
    );
}
void cev_domains() {
    namespace cev = model::equity::cev;
    std::array<cev::ModelParameters,1> models{{{1.f,0.f,0.f,.2f,.5f}}};
    const std::array<product::EuropeanOptionParameters,1> products{{{1.f,1U}}};
    const pg::PriceGradientConfiguration selection{{{"model.beta",{.002,pg::BumpScale::absolute}}}};
    auto prepare=[&] { return cev::prepare_cev_european_option_price_gradients(models,products,
        PriceConstruction::Aligned,{},selection); };
    require(first_order_task(prepare(), 0U, 0U).stencil.kind
                == pg::StencilKind::forward,
        "CEV beta lower boundary lost.");
    models[0].beta=.999f;
    require(first_order_task(prepare(), 0U, 0U).stencil.kind
                == pg::StencilKind::backward,
        "CEV beta upper boundary lost.");
    for (float beta : {.49f,1.f,std::numeric_limits<float>::quiet_NaN()}) {
        models[0].beta=beta; rejects(prepare);
    }
    models[0].beta=.7f; models[0].sigma=0.f; rejects(prepare);
}
void merton_domains() {
    namespace merton = model::equity::merton;
    const std::array<merton::ModelParameters,1> models{{{1.f,.03f,.01f,.2f,.5f,-.1f,.25f}}};
    const std::array<product::EuropeanOptionParameters,1> products{{{1.f,126U}}};
    const auto prepare=[&](const pg::PriceGradientConfiguration& selection) {
        return merton::prepare_merton_european_option_price_gradients(
            models,products,PriceConstruction::Aligned,{},selection);
    };
    const pg::PriceGradientConfiguration supported{{{"model.spot",{.005}},
        {"model.risk_free_rate",{.0005,pg::BumpScale::absolute}},
        {"model.dividend_yield",{.0005,pg::BumpScale::absolute}}, {"model.volatility",{.005}},
        {"model.jump_intensity",{.05,pg::BumpScale::absolute}},
        {"model.jump_log_mean",{.002,pg::BumpScale::absolute}},
        {"model.jump_log_volatility",{.005}}, {"product.strike",{.005}},
        {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
    const auto supported_plan = prepare(supported);
    require(supported_plan.sensitivity_count()==9U,
        "Merton supported coordinate count changed.");
    const auto intensity = first_order_task(supported_plan, 0U, 4U);
    require(intensity.nodes[1U].model.jump_intensity
                < intensity.nodes[0U].model.jump_intensity
            && intensity.nodes[2U].model.jump_intensity
                > intensity.nodes[0U].model.jump_intensity,
        "Merton jump-intensity endpoints were not prepared.");
    const auto maturity = first_order_task(supported_plan, 0U, 8U);
    require(maturity.nodes[1U].step_count == 251U
            && maturity.nodes[2U].step_count == 253U,
        "Merton coupled maturity endpoints changed calendar units.");
}
void heston_device_preparation() {
    namespace heston = model::equity::heston;
    const std::array<heston::ModelParameters, 2> models{{
        {1.f,0.f,0.f,0.f,1.5f,.04f,.3f,-.7f},
        {1.2f,0.f,0.f,.06f,.8f,.04f,.4f,1.f},
    }};
    const std::array<product::EuropeanOptionParameters, 2> products{{
        {1.f,16U}, {1.1f,12U}
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.initial_variance", {.001f, pg::BumpScale::absolute}},
        {"model.rho", {.002f, pg::BumpScale::absolute}},
        {"product.strike", {.005f}},
    }};
    const auto plan = heston::prepare_heston_european_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {},
        selection,
        {pg::SensitivityOrders::first_and_second}
    );
    require(
        plan.models.size() == models.size()
            && plan.products.size() == products.size()
            && plan.sensitivities.size() == selection.sensitivities.size(),
        "Heston compact sensitivity plan materialized row scenarios."
    );
    require(
        decltype(plan)::Preparation::parameter_owner(
                plan.sensitivities.front().parameter
            )
                == pg::SensitivityParameterOwner::model
            && decltype(plan)::Preparation::parameter_owner(
                plan.sensitivities.back().parameter
            )
                == pg::SensitivityParameterOwner::product,
        "Heston model/product sensitivity ownership was not resolved."
    );
    bool found_centered = false;
    bool found_one_sided = false;
    for (std::size_t row = 0U; row < plan.result_count; ++row) {
        typename decltype(plan)::Preparation::Scenario central{};
        require(
            decltype(plan)::Preparation::make_central(
                plan.models[row],
                equity::spot_relative::selected_central_product(
                    plan.models[row], plan.products[row],
                    plan.relative_quotes),
                plan.time, central
            ),
            "Heston central device row is invalid."
        );
        for (const auto& sensitivity : plan.sensitivities) {
            pg::SensitivityTask<
                typename decltype(plan)::Preparation::Scenario,
                4U
            > task{};
            int error = pg::device_preparation::valid;
            require(
                pg::device_preparation::
                    build_sensitivity_task<
                        pg::SensitivityOrders::first_and_second,
                        typename decltype(plan)::Preparation
                    >(
                        central,
                        sensitivity,
                        plan.time,
                        task,
                        error
                    ),
                "Heston host mirror of device preparation failed."
            );
            require(
                task.central_requirement == pg::CentralRequirement::payoff,
                "A diagonal Hessian did not retain its central payoff."
            );
            const auto node_count = pg::active_node_count(task.stencil);
            found_centered |= node_count == 3U;
            found_one_sided |= node_count == 4U;
            pg::SensitivityValues<4U> values{};
            const float x = task.stencil.parameter_values[0U];
            for (std::size_t node = 0U; node < node_count; ++node) {
                const float offset =
                    task.stencil.parameter_values[node] - x;
                values[node] = offset*offset;
            }
            const auto result = pg::reconstruct_sensitivity<
                pg::SensitivityOrders::first_and_second
            >(task.stencil, values);
            require(
                std::abs(result.first) < 2.e-4f,
                "Four-node first-derivative reconstruction is incorrect."
            );
            require(
                std::abs(result.second - 2.f *
                    std::pow(decltype(plan)::Preparation::coordinate_scale(
                        sensitivity.parameter, central
                    ), 2.f)) < 2.e-3f,
                "Four-node diagonal-Hessian reconstruction is incorrect."
            );
            for (std::size_t node = 0U; node < node_count; ++node) {
                values[node] = .7f;
            }
            require(
                pg::reconstruct_second_sensitivity(task.stencil, values)
                    == 0.0f,
                "A constant payoff produced a diagonal Hessian."
            );
        }
    }
    require(
        found_centered && found_one_sided,
        "Heston diagonal preparation did not cover both stencil shapes."
    );
    const std::array<product::EuropeanOptionParameters, 1> short_products{{
        {1.f, 1U}
    }};
    const auto short_plan = heston::prepare_heston_european_option_sensitivities(
        std::span<const heston::ModelParameters>(models.data(), 1U),
        short_products,
        PriceConstruction::Aligned,
        {},
        {{{"product.maturity_years",
           {2.f/504.f, pg::BumpScale::absolute}}}},
        {pg::SensitivityOrders::first_and_second}
    );
    typename decltype(short_plan)::Preparation::Scenario short_central{};
    require(decltype(short_plan)::Preparation::make_central(
                short_plan.models[0U],
                equity::spot_relative::selected_central_product(
                    short_plan.models[0U], short_plan.products[0U],
                    short_plan.relative_quotes),
                short_plan.time, short_central),
            "Short Heston maturity central row is invalid.");
    pg::SensitivityTask<
        typename decltype(short_plan)::Preparation::Scenario, 4U
    > maturity_task{};
    int maturity_error = pg::device_preparation::valid;
    require(pg::device_preparation::build_sensitivity_task<
                pg::SensitivityOrders::first_and_second,
                typename decltype(short_plan)::Preparation
            >(
                short_central, short_plan.sensitivities[0U],
                short_plan.time, maturity_task, maturity_error
            ),
            "Short Heston maturity diagonal preparation failed.");
    require(maturity_task.stencil.kind == pg::StencilKind::forward
                && pg::active_node_count(maturity_task.stencil) == 4U
                && maturity_task.nodes[3U].step_count == 8U,
            "Short maturity did not use four integer-grid nodes.");
    rejects([&] {
        heston::prepare_heston_european_option_price_gradients(
            models,
            products,
            PriceConstruction::Aligned,
            {},
            {{{"model.unknown", {.005f}}}}
        );
    });
}
void american_selection() {
    namespace heston = model::equity::heston;
    const std::array<heston::ModelParameters, 1> models{{
        {1.f, .03f, .01f, .04f, 1.5f, .04f, .4f, -.7f}
    }};
    const std::array<product::AmericanOptionParameters, 1> products{{
        {1.f, 20U, 7U}
    }};
    const auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return heston::prepare_heston_american_option_price_gradients(
            models, products, PriceConstruction::Aligned, {}, selection
        );
    };
    const auto plan = prepare({{{"product.strike", {.005}},
                                {"model.rho", {.002, pg::BumpScale::absolute}}}});
    require(plan.result_count == 1U && plan.models.size() == 1U
                && plan.products.size() == 1U
                && plan.sensitivities.size() == 2U,
            "American compact gradient shape is incorrect.");
    const auto strike = first_order_task(plan, 0U, 0U);
    const auto rho = first_order_task(plan, 0U, 1U);
    require(strike.nodes[1U].reuse_central
                && strike.nodes[2U].reuse_central,
            "American strike must reuse the central state trace.");
    require(!rho.nodes[1U].reuse_central && !rho.nodes[2U].reuse_central,
            "American dynamics bump incorrectly reuses the central state.");
    rejects([&] { prepare({{{"product.maturity_years",
                            {1.f / 504.f, pg::BumpScale::absolute}}}}); });
    rejects([&] { prepare({{{"product.exercise_interval_days",
                            {1.f, pg::BumpScale::absolute}}}}); });
}
void spot_relative_contract() {
    namespace quotes = equity::spot_relative;
    const std::array<bs::ModelParameters, 1> models{{
        {3.f, 0.f, 0.f, .2f}
    }};
    const std::array<product::EuropeanOptionParameters, 1> products{{
        {.9f, 252U}
    }};
    const pg::PriceGradientConfiguration selected{{
        {"model.spot", {.005, pg::BumpScale::relative}},
        {"product.strike", {.01, pg::BumpScale::absolute}},
    }};
    const auto plan = bs::prepare_black_scholes_european_option_price_gradients(
        models, products, PriceConstruction::Aligned, {}, selected
    );
    require(plan.products[0].strike == .9f && plan.relative_quotes,
        "Quoted strike was not retained in its source dataset.");
    const auto spot = first_order_task(plan, 0U, 0U);
    const float frozen_strike = central_row(plan, 0U).product.strike;
    require(spot.nodes[1U].product.strike == frozen_strike
            && spot.nodes[2U].product.strike == frozen_strike,
        "Spot bump moved the frozen strike.");
    const auto strike = first_order_task(plan, 0U, 1U);
    require(std::abs(strike.nodes[2U].product.strike
                     - strike.nodes[1U].product.strike - .06f) < 2.e-6f,
        "Absolute ratio bump did not scale by central spot.");
    pg::SensitivityValues<3U> values{};
    for (std::size_t node = 0U; node < 3U; ++node) {
        const float absolute = strike.nodes[node].product.strike;
        values[node] = absolute * absolute;
    }
    require(std::abs(pg::reconstruct_first_sensitivity(
                strike.stencil, values
            ) - 2.f * .9f * 3.f * 3.f) < 2.e-4f,
        "Strike stencil does not differentiate in quoted-ratio units.");

    const std::vector<bs::ModelParameters> equal_spot_models{
        models.front(), models.front()
    };
    const std::vector<product::EuropeanOptionParameters> equal_spot_quotes{
        products.front(), {.8f, 252U}
    };
    const auto compact_cartesian = quotes::resolve_rows(
        equal_spot_models, equal_spot_quotes,
        PriceConstruction::CartesianProduct
    );
    require(compact_cartesian.construction == PriceConstruction::CartesianProduct
            && compact_cartesian.models.size() == 2U
            && compact_cartesian.products.size() == 2U
            && std::abs(compact_cartesian.products[1U].strike - 2.4f)
                < 1.e-6f,
        "Common-spot Cartesian inputs were expanded unnecessarily.");

    const std::vector<bs::ModelParameters> varying_spot_models{
        models.front(), {1.f, 0.f, 0.f, .2f}
    };
    const std::vector<product::RangeAccrualParameters> relative_ranges{
        {252U, 21U, .8f, 1.2f, .05f}
    };
    const auto compact_relative = quotes::resolve_rows(
        varying_spot_models, relative_ranges,
        PriceConstruction::CartesianProduct
    );
    require(compact_relative.construction == PriceConstruction::CartesianProduct
            && compact_relative.models.size() == 2U
            && compact_relative.products.size() == 1U,
        "Already-relative Cartesian product was expanded unnecessarily.");

    const auto barrier = quotes::resolve_product(
        product::DoubleKnockOutOptionParameters{1.f, .9f, 1.1f, 252U},
        3.f
    );
    require(barrier.lower_barrier < 3.f && barrier.upper_barrier > 3.f
            && std::abs(barrier.strike - 3.f) < 1.e-6f,
        "Double barrier does not adapt to nonunit spot.");
    const std::array<product::DoubleKnockOutOptionParameters, 1> barriers{{
        {1.f, .9f, 1.1f, 252U}
    }};
    const auto barrier_plan =
        bs::prepare_black_scholes_double_knock_out_option_price_gradients(
            models, barriers, PriceConstruction::Aligned, {},
            {{{"model.spot", {.005f}},
              {"product.lower_barrier", {.01f, pg::BumpScale::absolute}}}}
        );
    const auto spot_barrier_task = first_order_task(barrier_plan, 0U, 0U);
    const auto lower_barrier_task = first_order_task(barrier_plan, 0U, 1U);
    const float frozen_lower = central_row(
        barrier_plan, 0U).product.lower_barrier;
    require(barrier_plan.products[0].lower_barrier == .9f
            && spot_barrier_task.nodes[1U].product.lower_barrier
                == frozen_lower
            && spot_barrier_task.nodes[2U].product.lower_barrier
                == frozen_lower
            && std::abs(lower_barrier_task.nodes[2U].product.lower_barrier
                - lower_barrier_task.nodes[1U].product.lower_barrier
                - .06f) < 2.e-6f,
        "Spot and barrier-ratio bumps were not independent.");
    const auto touch = quotes::resolve_product(
        product::UpOneTouchParameters{1.1f, .25f, 252U}, 3.f
    );
    require(std::abs(touch.barrier - 3.3f) < 1.e-6f
            && touch.cash_payoff == .25f,
        "Touch cash payoff was scaled with its barrier.");
    const auto range = quotes::resolve_product(
        product::RangeAccrualParameters{252U, 21U, .9f, 1.1f, .05f},
        3.f
    );
    require(range.lower_barrier == .9f && range.upper_barrier == 1.1f,
        "Already-relative range accrual was scaled twice.");
    const std::array<product::RangeAccrualParameters, 1> range_quotes{{
        {252U, 21U, .9f, 1.1f, .05f}
    }};
    const auto range_plan =
        bs::prepare_black_scholes_range_accrual_price_gradients(
            models, range_quotes, PriceConstruction::Aligned, {},
            {{{"model.spot", {.005f}}}}
        );
    const auto range_spot = first_order_task(range_plan, 0U, 0U);
    require(!range_plan.relative_quotes
            && range_spot.nodes[1U].product.lower_barrier == .9f
            && range_spot.nodes[2U].product.lower_barrier == .9f
            && range_spot.nodes[1U].product.upper_barrier == 1.1f
            && range_spot.nodes[2U].product.upper_barrier == 1.1f,
        "Spot bump changed intrinsic normalized range bounds.");
    const auto autocall = quotes::resolve_product(
        product::AthenaAutocallParameters{252U, 21U, 1.f, .7f, .06f, 1.0f},
        3.f
    );
    require(autocall.autocall_barrier == 3.f
            && std::abs(autocall.protection_barrier - 2.1f) < 1.e-6f
            && autocall.reference_spot == 3.f,
        "Autocall level/reference resolution changed.");
}
void fixed_income_selection() {
    namespace cir = model::fixed_income::cir;
    const std::array<cir::ModelParameters, 2> models{{
        {{.5f, .04f, .1f}, .03f},
        {{.2f, .02f, .05f}, 0.f},
    }};
    const std::array<product::RegularEuropeanSwaptionParameters, 2> products{{
        {1.f, .03f, .5f, 252U, 126U, 4U},
        {2.f, .04f, .25f, 126U, 63U, 8U},
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.mean_reversion", {.005}},
        {"model.initial_state", {.0005, pg::BumpScale::absolute}},
        {"product.strike", {.0001, pg::BumpScale::absolute}},
        {"product.maturity_years",
         {1.f / 504.f, pg::BumpScale::absolute}},
    }};
    const auto plan = cir::prepare_cir_european_swaption_price_gradients(
        models, products, PriceConstruction::CartesianProduct, {}, selection
    );
    require(plan.result_count == 4U && plan.models.size() == 2U
        && plan.products.size() == 2U && plan.sensitivities.size() == 4U
        && plan.maximum_payment_count == 8U,
        "Fixed-income selected Cartesian shape is incorrect.");
    const auto mean_reversion = first_order_task(plan, 0U, 0U);
    const auto initial_state = first_order_task(plan, 0U, 1U);
    const auto strike = first_order_task(plan, 0U, 2U);
    const auto maturity = first_order_task(plan, 0U, 3U);
    require(mean_reversion.nodes[1U].model.process.mean_reversion
            < mean_reversion.nodes[0U].model.process.mean_reversion
        && initial_state.nodes[1U].model.initial_state
            < initial_state.nodes[0U].model.initial_state
        && strike.nodes[1U].product.strike
            < strike.nodes[0U].product.strike,
        "Fixed-income nested/product coordinate resolution changed.");
    require(
        maturity.nodes[1U].step_count + 2U
                == maturity.nodes[2U].step_count
            && maturity.nodes[1U].reuse_central
            && maturity.nodes[2U].reuse_central
            && maturity.nodes[1U].product.payment_count
                == maturity.nodes[0U].product.payment_count
            && decltype(plan)::Preparation::parameter_owner(
                plan.sensitivities[3U].parameter
            ) == pg::SensitivityParameterOwner::maturity,
        "Fixed-income terminal date did not move independently of its calendar."
    );
    require(first_order_task(plan, 2U, 1U).stencil.kind
                == pg::StencilKind::forward,
        "Fixed-income zero initial-state boundary was not one-sided.");
    rejects([&] {
        cir::prepare_cir_european_swaption_price_gradients(
            models, products, PriceConstruction::Aligned, {},
            pg::PriceGradientConfiguration{{
                {"product.exercise_time_days",
                 {1., pg::BumpScale::absolute}}
            }}
        );
    });
}
}  // namespace
int main() {
    try { stencils(); covariance(); selection(); cev_domains(); merton_domains();
        heston_device_preparation();
        american_selection(); spot_relative_contract(); fixed_income_selection();
        std::cout << "Gradient host contracts passed\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
