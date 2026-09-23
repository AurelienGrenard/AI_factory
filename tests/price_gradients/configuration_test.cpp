// Independent polynomial, domain, selection and Brownian covariance checks for host plans.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/cev/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"
#include "common/price_gradients/result.cuh"
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
        const auto weights = equity::price_gradients::brownian_endpoint_weights(dates[0], dates[1], dates[2]);
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
    require(plan.result_count == 4U && plan.scenarios.size() == 20U && plan.stencils.size() == 8U,
        "Selected Cartesian shape is incorrect.");
    for (const auto& row : plan.scenarios) require(row.model.risk_free_rate == 0.f && row.model.dividend_yield == 0.f,
        "Unselected zero parameters were changed.");
    require(plan.scenarios[5].product.strike == .9f && plan.scenarios[10].model.spot == 1.2f, "Cartesian ordering changed.");
    require(plan.scenarios[1].central_requirement == pg::CentralRequirement::state
        && plan.scenarios[3].central_requirement == pg::CentralRequirement::state,
        "Homogeneous spot/strike should reuse state without central payoff.");
    const auto volatility = prepare(pg::PriceGradientConfiguration{{{"model.volatility", {.01f}}}});
    require(volatility.scenarios[1].central_requirement == pg::CentralRequirement::none,
        "Centered dynamics gradient incorrectly requires central state.");
    const auto boundary = prepare(pg::PriceGradientConfiguration{{{"model.volatility", {.3f, pg::BumpScale::absolute}}}});
    require(boundary.stencils[0].kind != pg::StencilKind::centered
        && boundary.scenarios[1].central_requirement == pg::CentralRequirement::payoff,
        "Boundary stencil does not declare its central payoff dependency.");
    require(prepare(pg::PriceGradientConfiguration{}).scenarios.size() == 4U, "Empty selection does not produce price-only scenarios.");
    rejects([&] { prepare(pg::PriceGradientConfiguration{{{"model.spot", {.01f}}, {"model.spot", {.02f}}}}); });
    rejects([&] { prepare(pg::PriceGradientConfiguration{{{"model.jump_intensity", {.01f}}}}); });
    const auto dates = prepare(pg::PriceGradientConfiguration{{{"product.maturity_years", {1.f/504.f, pg::BumpScale::absolute}}}});
    require(dates.scenarios[1].step_count == 503U && dates.scenarios[2].step_count == 505U, "One-step maturity bump changed calendar units.");
    rejects([&] { prepare(pg::PriceGradientConfiguration{{{"product.maturity_years", {.5f/504.f, pg::BumpScale::absolute}}}}); });
}
void cev_domains() {
    namespace cev = model::equity::cev;
    std::array<cev::ModelParameters,1> models{{{1.f,0.f,0.f,.2f,.5f}}};
    const std::array<product::EuropeanOptionParameters,1> products{{{1.f,1U}}};
    const pg::PriceGradientConfiguration selection{{{"model.beta",{.002,pg::BumpScale::absolute}}}};
    auto prepare=[&] { return cev::prepare_cev_european_option_price_gradients(models,products,
        PriceConstruction::Aligned,{},selection); };
    require(prepare().stencils[0].kind==pg::StencilKind::forward,"CEV beta lower boundary lost.");
    models[0].beta=.999f;
    require(prepare().stencils[0].kind==pg::StencilKind::backward,"CEV beta upper boundary lost.");
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
        {"model.jump_log_mean",{.002,pg::BumpScale::absolute}},
        {"model.jump_log_volatility",{.005}}, {"product.strike",{.005}}}};
    require(prepare(supported).sensitivity_count()==7U,"Merton supported coordinate count changed.");
    rejects([&] { prepare(pg::PriceGradientConfiguration{{
        {"model.jump_intensity",{.001,pg::BumpScale::absolute}}}}); });
    rejects([&] { prepare(pg::PriceGradientConfiguration{{
        {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}}); });
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
    require(plan.scenarios.size() == 5U && plan.stencils.size() == 2U,
            "American gradient scenario shape is incorrect.");
    require(plan.scenarios[1].reuse_central && plan.scenarios[2].reuse_central,
            "American strike must reuse the central state trace.");
    require(!plan.scenarios[3].reuse_central && !plan.scenarios[4].reuse_central,
            "American dynamics bump incorrectly reuses the central state.");
    rejects([&] { prepare({{{"product.maturity_years",
                            {1.f / 504.f, pg::BumpScale::absolute}}}}); });
    rejects([&] { prepare({{{"product.exercise_interval_days",
                            {1.f, pg::BumpScale::absolute}}}}); });
}
}  // namespace
int main() {
    try { stencils(); covariance(); selection(); cev_domains(); merton_domains();
        american_selection();
        std::cout << "Gradient host contracts passed\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
