// Every short-rate family can discount the same equity basket through R.
#include "correlation/dataset.hpp"
#include "multi_asset/rate_adjusted_model_group.cuh"
#include "curve/flat/term_structure_impl.cuh"
#include "curve/nelson_siegel/term_structure_impl.cuh"
#include "curve/svensson/term_structure_impl.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "model/fixed_income/vasicek/correlated_dynamics.cuh"
#include "model/fixed_income/constant_rate/correlated_dynamics.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/correlated_dynamics.cuh"
#include "model/fixed_income/cir/correlated_dynamics.cuh"
#include "model/fixed_income/hull_white/correlated_dynamics.cuh"
#include "model/fixed_income/cir_plus_plus/correlated_dynamics.cuh"
#include "model/fixed_income/g2/correlated_dynamics.cuh"
#include "model/fixed_income/g2_plus_plus/correlated_dynamics.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/cross_asset_price_generation.cuh"

#include <bit>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace {
using namespace ai_factory::workbench;
using BS=model::equity::black_scholes::DynamicsPolicy;
using Pack=simulation::MultiAssetPack<simulation::RateAdjustedModelGroup<BS,2U>>;
using NoRough=simulation::MultiRoughPack<>;
using Worst=product::WorstOfCallPathPolicy;
using Flat=curve::flat::AnalyticsProvider;
using NS=curve::nelson_siegel::AnalyticsProvider;
using Svensson=curve::svensson::AnalyticsProvider;
using Expression=RowIndexExpression;

template<typename Policy,unsigned TeamSize=1U>
float price(const typename Policy::Parameters& rate,
            const std::vector<typename Policy::CurveParameters>& curves={},
            unsigned curve_index=0U,float strike=0.0f) {
    offline::pricing::MultiAssetHostModels<Pack> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    std::vector<std::string> names{"a.spot","b.spot","rate.x"};
    const unsigned n=2U+Policy::kBrownianDimension;
    if constexpr (Policy::kBrownianDimension==2U) names.push_back("rate.y");
    std::vector<double> dense(n*n,0.0);
    for(unsigned i=0;i<n;++i) dense[i*n+i]=1.0;
    dense[0*n+2]=dense[2*n+0]=0.2;
    dense[1*n+2]=dense[2*n+1]=-0.1;
    if constexpr (Policy::kBrownianDimension==2U)
        dense[2*n+3]=dense[3*n+2]=Policy::internal_correlation(rate);
    const correlation::MatrixRow matrix{
        "rate",names,correlation::packed_cholesky(dense,n)};
    std::vector<typename Policy::CurveParameters> selected_curves;
    if constexpr (Policy::kUsesCurve) selected_curves={curves.at(curve_index)};
    std::vector<Expression> slots;
    for(std::size_t slot=0U;slot<5U+(Policy::kUsesCurve?1U:0U);++slot)
        slots.push_back(Expression::source(slot,1U));
    const auto plan=compile_row_index_plan(Expression::aligned(slots),
        slots.size());
    const auto result=offline::pricing::execute_cross_asset_prices<
        Pack,NoRough,Policy,Worst,TeamSize>(
        equities,rough,{rate},{{strike,30U}},{matrix},names,names,
        plan,512U,512U,1.0f/252.0f,1U,32U,1597U,
        selected_curves);
    if(!std::isfinite(result.prices[0]))
        throw std::runtime_error("non-finite cross-asset price");
    return result.prices[0];
}
void require_close(float actual,float expected,float tolerance,const char* msg) {
    if(std::abs(actual-expected)>tolerance)
        throw std::runtime_error(msg);
}
void all_rate_families() {
    using Vasicek=model::fixed_income::vasicek::CorrelatedRatePolicy;
    using OU=model::fixed_income::ornstein_uhlenbeck::CorrelatedRatePolicy;
    using CIR=model::fixed_income::cir::CorrelatedRatePolicy;
    using HW=model::fixed_income::hull_white::CorrelatedRatePolicy<Flat>;
    using CIRpp=model::fixed_income::cir_plus_plus::CorrelatedRatePolicy<Flat>;
    using G2=model::fixed_income::g2::CorrelatedRatePolicy;
    using G2pp=model::fixed_income::g2_plus_plus::CorrelatedRatePolicy<Flat>;
    using HWNS=model::fixed_income::hull_white::CorrelatedRatePolicy<NS>;
    using CIRppS=model::fixed_income::cir_plus_plus::CorrelatedRatePolicy<Svensson>;
    using G2ppS=model::fixed_income::g2_plus_plus::CorrelatedRatePolicy<Svensson>;
    require_close(price<Vasicek>({{0.6f,0.025f,0.04f},0.02f}),
        1.0f,4.0e-5f,"Vasicek basket discount");
    require_close(price<OU>({{0.6f,0.04f},0.02f}),
        1.0f,4.0e-5f,"OU basket discount");
    require_close(price<CIR>({{0.6f,0.025f,0.08f},0.02f}),
        1.0f,4.0e-5f,"CIR basket discount");
    const CIR::Parameters cir_rate{{0.6f,0.025f,0.08f},0.02f};
    require_close(price<CIR,4U>(cir_rate,{},0U,1.0f),
        price<CIR>(cir_rate,{},0U,1.0f),2.0e-6f,
        "CIR rate depends on TeamSize");

    require_close(price<HW>({0.6f,0.04f},{{0.01f},{0.06f}},1U),
        1.0f,4.0e-5f,"Hull-White basket discount");
    require_close(price<CIRpp>({{0.6f,0.025f,0.08f},0.02f},
        {{0.01f},{0.06f}},1U),1.0f,4.0e-5f,"CIR++ basket discount");
    require_close(price<G2>({{0.6f,0.04f,0.9f,0.03f,0.3f},
        {0.02f,0.01f}}),1.0f,4.0e-5f,"G2 basket discount");
    const G2::Parameters g2_rate{
        {0.6f,0.04f,0.9f,0.03f,0.3f},{0.02f,0.01f}};
    require_close(price<G2,4U>(g2_rate,{},0U,1.0f),
        price<G2>(g2_rate,{},0U,1.0f),2.0e-6f,
        "G2 two-factor rate depends on TeamSize");

    require_close(price<G2pp>({{0.6f,0.04f,0.9f,0.03f,0.3f}},
        {{0.01f},{0.06f}},1U),1.0f,4.0e-5f,
        "G2++ basket discount");
    constexpr float maturity=30.0f/252.0f;
    const float expected=1.0f-std::exp(-0.06f*maturity);
    require_close(price<HW>({0.6f,0.0f},{{0.01f},{0.06f}},1U,1.0f),
        expected,6.0e-5f,"Hull-White curve index or shift");
    require_close(price<G2pp>({{0.6f,0.0f,0.9f,0.0f,0.3f}},
        {{0.01f},{0.06f}},1U,1.0f),
        expected,6.0e-5f,"G2++ curve index or shift");
    require_close(price<HWNS>({0.6f,0.0f},
        {{0.01f,0.0f,0.0f,1.0f},{0.06f,0.0f,0.0f,1.0f}},
        1U,1.0f),expected,6.0e-5f,"Nelson-Siegel curve shift");
    require_close(price<CIRppS>({{0.6f,0.025f,0.08f},0.02f},
        {{0.01f,0.0f,0.0f,0.0f,1.0f,2.0f},
         {0.06f,0.0f,0.0f,0.0f,1.0f,2.0f}},
        1U),1.0f,4.0e-5f,"Svensson CIR++ basket discount");
    require_close(price<G2ppS>({{0.6f,0.0f,0.9f,0.0f,0.3f}},
        {{0.01f,0.0f,0.0f,0.0f,1.0f,2.0f},
         {0.06f,0.0f,0.0f,0.0f,1.0f,2.0f}},
        1U,1.0f),expected,6.0e-5f,"Svensson G2++ curve shift");
}
void compact_rate_and_curve_selection() {
    using Vasicek=model::fixed_income::vasicek::CorrelatedRatePolicy;
    using HW=model::fixed_income::hull_white::CorrelatedRatePolicy<Flat>;
    offline::pricing::MultiAssetHostModels<Pack> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> names{"a.spot","b.spot","rate.x"};
    const std::vector<correlation::MatrixRow> matrices{{
        "identity",names,correlation::packed_cholesky({
            1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0},3U)}};
    const auto rate_plan=compile_row_index_plan(Expression::cartesian({
        Expression::aligned({Expression::source(0U,1U),
            Expression::source(1U,1U),Expression::source(3U,1U),
            Expression::source(4U,1U)}),
        Expression::source(2U,2U)}),5U);
    const std::vector<Vasicek::Parameters> rates{
        {{0.6f,0.01f,0.0f},0.01f},
        {{0.6f,0.06f,0.0f},0.06f}};
    const auto selected_rates=offline::pricing::execute_cross_asset_prices<
        Pack,NoRough,Vasicek,Worst>(equities,rough,rates,
            {{1.0f,30U}},matrices,names,names,rate_plan,
            128U,128U,1.0f/252.0f,1U,32U,1717U);
    constexpr float maturity=30.0f/252.0f;
    for(unsigned i=0U;i<2U;++i) {
        const float rate=i==0U?0.01f:0.06f;
        require_close(selected_rates.prices[i],
            1.0f-std::exp(-rate*maturity),6e-5f,
            "compact rate selector mismatch");
    }
    const auto curve_plan=compile_row_index_plan(Expression::cartesian({
        Expression::aligned({Expression::source(0U,1U),
            Expression::source(1U,1U),Expression::source(2U,1U),
            Expression::source(4U,1U)}),
        Expression::source(3U,2U),Expression::source(5U,2U)}),6U);
    const auto selected_curves=offline::pricing::execute_cross_asset_prices<
        Pack,NoRough,HW,Worst>(equities,rough,{{0.6f,0.0f}},
            {{1.0f,30U},{0.9f,30U}},matrices,names,names,
            curve_plan,128U,128U,1.0f/252.0f,1U,32U,1717U,
            {{0.01f},{0.06f}});
    // Last child varies fastest: each product appears with both curves.
    for(unsigned product=0U;product<2U;++product)
        for(unsigned curve=0U;curve<2U;++curve) {
            const float strike=product==0U?1.0f:0.9f;
            const float rate=curve==0U?0.01f:0.06f;
            require_close(selected_curves.prices[product*2U+curve],
                1.0f-strike*std::exp(-rate*maturity),6e-5f,
                "compact curve or product selector mismatch");
        }
}
} // namespace

int main() {
    all_rate_families();
    compact_rate_and_curve_selection();
}
