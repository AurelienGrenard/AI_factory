// Every short-rate family can discount the same equity basket through R.
#include "correlation/dataset.hpp"
#include "common/simulation/rate_adjusted_equity_path.cuh"
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
using Heston=model::equity::rough_heston::CausalFftPathPolicy;

template<typename Policy,unsigned TeamSize=1U>
float price(const typename Policy::Parameters& rate,
            const std::vector<typename Policy::CurveParameters>& curves={},
            unsigned curve_index=0U,float strike=0.0f) {
    offline::pricing::MultiAssetHostModels<Pack> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    equities.scenario_indices={0U,0U};
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
    const auto result=offline::pricing::execute_cross_asset_prices<
        Pack,NoRough,Policy,Worst,TeamSize>(
        equities,rough,{rate},{0U},1U,
        {{strike,30U}},{matrix},names,names,
        {{0U,0U,0U,curve_index,1.0f}},512U,512U,
        1.0f/252.0f,1U,32U,1597U,curves);
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
void constant_rate_team() {
    using Constant=model::fixed_income::constant_rate::CorrelatedRatePolicy;
    offline::pricing::MultiAssetHostModels<Pack> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f}};
    equities.scenario_indices={0U,0U};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> names{"a.spot","b.spot"};
    const correlation::MatrixRow matrix{
        "constant",names,correlation::packed_cholesky({
            1.0,0.35,0.35,1.0},2U)};
    const auto run=[&]<unsigned TeamSize>() {
        return offline::pricing::execute_cross_asset_prices<
            Pack,NoRough,Constant,Worst,TeamSize>(
                equities,rough,{{0.02f}},{0U},1U,
                {{0.85f,21U}},{matrix},names,names,
                {{0U,0U,0U,0U,1.0f}},512U,512U,
                1.0f/252.0f,1U,32U,1933U);
    };
    const auto one=run.template operator()<1U>();
    const auto four=run.template operator()<4U>();
    if(std::abs(one.prices[0]-four.prices[0])>2.0e-6f
        ||std::abs(one.standard_errors[0]
            -four.standard_errors[0])>2.0e-6f)
        throw std::runtime_error(
            "constant short rate depends on TeamSize");
}

void g2_internal_rho_rejected() {
    using G2=model::fixed_income::g2::CorrelatedRatePolicy;
    offline::pricing::MultiAssetHostModels<Pack> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    equities.scenario_indices={0U,0U};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> names{"a.spot","b.spot","rate.x","rate.y"};
    const correlation::MatrixRow matrix{
        "wrong_rho",names,correlation::packed_cholesky({
            1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,0.1,
            0.0,0.0,0.1,1.0
        },4U)};
    bool rejected=false;
    try {
        (void)offline::pricing::execute_cross_asset_prices<
            Pack,NoRough,G2,Worst>(equities,rough,
            {{{0.6f,0.04f,0.9f,0.03f,0.3f},{0.02f,0.01f}}},
            {0U},1U,{{0.0f,4U}},{matrix},names,names,
            {{0U,0U,0U,0U,1.0f}},128U,128U,
            1.0f/252.0f,1U,32U,17U);
    } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("G2 matrix/model rho mismatch accepted");
}
void g2_gaussian_integral() {
    using G2=model::fixed_income::g2::CorrelatedRatePolicy;
    offline::pricing::MultiAssetHostModels<Pack> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    equities.scenario_indices={0U,0U};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> names{"a.spot","b.spot","rate.x","rate.y"};
    constexpr double rho=0.65,ax=0.7,ay=1.1,sx=0.18,sy=0.14;
    const correlation::MatrixRow matrix{
        "g2",names,correlation::packed_cholesky({
            1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,rho,
            0.0,0.0,rho,1.0
        },4U)};
    const auto run=offline::pricing::execute_cross_asset_prices<
        Pack,NoRough,G2,Worst>(equities,rough,
        {{{static_cast<float>(ax),static_cast<float>(sx),
           static_cast<float>(ay),static_cast<float>(sy),
           static_cast<float>(rho)},{0.0f,0.0f}}},
        {0U},1U,{{1.0f,4U}},{matrix},names,names,
        {{0U,0U,0U,0U,1.0f}},32768U,32768U,
        0.25f,1U,32U,39171U);
    // I=integral_0^1 r_t dt is centered Gaussian. Compute its variance
    // independently from the Brownian-kernel integral.
    constexpr unsigned cells=16384U;
    double variance=0.0;
    for(unsigned i=0;i<cells;++i) {
        const double u=(i+0.5)/cells;
        const double gx=-std::expm1(-ax*u)/ax;
        const double gy=-std::expm1(-ay*u)/ay;
        variance+=(sx*sx*gx*gx+sy*sy*gy*gy
            +2.0*rho*sx*sy*gx*gy)/cells;
    }
    const double sd=std::sqrt(variance);
    const auto cdf=[](double z) {
        return 0.5*std::erfc(-z/std::sqrt(2.0));
    };
    const double reference=0.5-std::exp(variance/2.0)*cdf(-sd);
    if(std::abs(run.prices[0]-reference)
        >5.0*run.standard_errors[0]+2.0e-4)
        throw std::runtime_error("G2 integrated Gaussian law incorrect");
}
template<typename Policy>
double benchmark_large_basket(
    const typename Policy::Parameters& rate,
    const std::vector<typename Policy::CurveParameters>& curves={}) {
    using Large=simulation::MultiAssetPack<
        simulation::RateAdjustedModelGroup<BS,30U>>;
    offline::pricing::MultiAssetHostModels<Large> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f}};
    equities.scenario_indices.assign(30U,0U);
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    constexpr unsigned n=30U+Policy::kBrownianDimension;
    std::vector<std::string> names;
    std::vector<double> dense(n*n,0.0);
    for(unsigned i=0;i<n;++i) {
        dense[i*n+i]=1.0;
        names.push_back("factor_"+std::to_string(i));
    }
    dense[0*n+30]=dense[30*n+0]=0.2;
    if constexpr (Policy::kBrownianDimension==2U)
        dense[30*n+31]=dense[31*n+30]
            =Policy::internal_correlation(rate);
    const correlation::MatrixRow matrix{
        "large",names,correlation::packed_cholesky(dense,n)};
    std::vector<double> times;
    for(unsigned trial=0;trial<3;++trial) {
        const auto run=offline::pricing::execute_cross_asset_prices<
            Large,NoRough,Policy,Worst>(equities,rough,{rate},{0U},1U,
            {{0.9f,21U}},{matrix},names,names,
            {{0U,0U,0U,0U,1.0f}},8192U,8192U,
            1.0f/252.0f,1U,32U,5893U,curves);
        if(!std::isfinite(run.prices[0]))
            throw std::runtime_error("large cross-asset price invalid");
        times.push_back(run.kernel_seconds*1000.0);
    }
    std::sort(times.begin(),times.end());
    return times[1];
}
void large_shared_matrix() {
    using OU=model::fixed_income::ornstein_uhlenbeck::CorrelatedRatePolicy;
    using Large=simulation::MultiAssetPack<
        simulation::RateAdjustedModelGroup<BS,160U>>;
    offline::pricing::MultiAssetHostModels<Large> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    equities.scenario_indices.assign(160U,0U);
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    constexpr unsigned n=161U;
    std::vector<std::string> names;
    std::vector<double> dense(n*n,0.0);
    for(unsigned i=0;i<n;++i) {
        dense[i*n+i]=1.0;
        names.push_back("factor_"+std::to_string(i));
    }
    dense[0*n+160]=dense[160*n+0]=0.2;
    const correlation::MatrixRow matrix{
        "large_shared",names,correlation::packed_cholesky(dense,n)};
    const auto run=offline::pricing::execute_cross_asset_prices<
        Large,NoRough,OU,Worst>(equities,rough,{{{0.6f,0.0f},0.02f}},
        {0U},1U,{{0.0f,1U}},{matrix},names,names,
        {{0U,0U,0U,0U,1.0f}},128U,128U,
        1.0f/252.0f,1U,32U,8727U);
    require_close(run.prices[0],1.0f,5.0e-5f,
        "large Cholesky shared-memory launch");

    equities.models[0].volatility=0.2f;
    const std::vector<OU::Parameters> stochastic_rates{
        {{0.6f,0.04f},0.02f}};
    const std::vector<Worst::ProductParameters> products{
        {0.7f,21U}};
    constexpr std::size_t paths=32768U;
    auto measure=[&]<unsigned TeamSize>() {
        std::vector<double> milliseconds;
        float price=0.0f,error=0.0f;
        for(unsigned trial=0U;trial<3U;++trial) {
            const auto result=offline::pricing::execute_cross_asset_prices<
                Large,NoRough,OU,Worst,TeamSize>(
                    equities,rough,stochastic_rates,{0U},1U,
                    products,{matrix},names,names,
                    {{0U,0U,0U,0U,1.0f}},paths,paths,
                    1.0f/252.0f,1U,32U,8727U);
            price=result.prices[0];
            error=result.standard_errors[0];
            milliseconds.push_back(result.kernel_seconds*1000.0);
        }
        std::sort(milliseconds.begin(),milliseconds.end());
        return std::pair{milliseconds[1U],std::pair{price,error}};
    };
    const auto team1=measure.template operator()<1U>();
    const auto team2=measure.template operator()<2U>();
    const auto team4=measure.template operator()<4U>();
    const auto team8=measure.template operator()<8U>();
    const auto team16=measure.template operator()<16U>();
    const auto team32=measure.template operator()<32U>();
    for(const auto* result:{&team2,&team4,&team8,&team16,&team32}) {
        if(std::abs(result->second.first-team1.second.first)>2.0e-6f
            ||std::abs(result->second.second-team1.second.second)>2.0e-6f)
            throw std::runtime_error(
                "160-asset result depends on TeamSize");
    }
    std::cout<<"D=160+rate paths="<<paths
        <<" team1_ms="<<team1.first
        <<" team2_ms="<<team2.first
        <<" team4_ms="<<team4.first
        <<" team8_ms="<<team8.first
        <<" team16_ms="<<team16.first
        <<" team32_ms="<<team32.first<<'\n';
}
void benchmark_rate_families() {
    using V=model::fixed_income::vasicek::CorrelatedRatePolicy;
    using C=model::fixed_income::cir::CorrelatedRatePolicy;
    using H=model::fixed_income::hull_white::CorrelatedRatePolicy<Flat>;
    using G=model::fixed_income::g2::CorrelatedRatePolicy;
    using GP=model::fixed_income::g2_plus_plus::CorrelatedRatePolicy<Flat>;
    std::cout<<"D=30+rate median_ms Vasicek="
        <<benchmark_large_basket<V>({{0.6f,0.02f,0.03f},0.02f})
        <<" CIR="
        <<benchmark_large_basket<C>({{0.6f,0.02f,0.03f},0.02f})
        <<" HullWhite="
        <<benchmark_large_basket<H>({0.6f,0.03f},{{0.02f}})
        <<" G2="
        <<benchmark_large_basket<G>({{0.6f,0.03f,0.9f,0.02f,0.3f},
                                     {0.02f,0.01f}})
        <<" G2++="
        <<benchmark_large_basket<GP>({{0.6f,0.03f,0.9f,0.02f,0.3f}},
                                      {{0.02f}})
        <<'\n';
}
void g2_with_rough() {
    using G2=model::fixed_income::g2::CorrelatedRatePolicy;
    using OneBS=simulation::MultiAssetPack<simulation::RateAdjustedModelGroup<BS,1U>>;
    using OneRough=simulation::MultiRoughPack<Heston>;
    offline::pricing::MultiAssetHostModels<OneBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    equities.scenario_indices={0U};
    offline::pricing::MultiRoughHostModels<OneRough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.0f,2.0f,0.0f,0.0f,0.12f,-0.4f}};
    rough.scenario_indices={0U};
    const std::vector<std::string> external{
        "bs.spot","rough.spot","rough.vol","rate.x","rate.y"};
    const std::vector<std::string> internal{
        "rough.vol","bs.spot","rate.x","rate.y","rough.spot"};
    const correlation::MatrixRow ordered{
        "rough_g2",internal,correlation::packed_cholesky({
            1.0,0.0,0.0,0.0,-0.4,
            0.0,1.0,0.2,0.0,0.0,
            0.0,0.2,1.0,0.3,0.0,
            0.0,0.0,0.3,1.0,0.1,
            -0.4,0.0,0.0,0.1,1.0
        },5U)};
    const auto matrix=correlation::reorder_factors(ordered,external);
    const auto result=offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,G2,Worst>(equities,rough,
        {{{0.6f,0.04f,0.9f,0.03f,0.3f},{0.02f,0.01f}}},
        {0U},1U,{{0.0f,8U}},{matrix},external,internal,
        {{0U,0U,0U,0U,1.0f}},128U,128U,
        1.0f/252.0f,1U,32U,2041U);
    require_close(result.prices[0],1.0f,5.0e-5f,
        "rough basket with correlated two-factor rate");
    const auto team4=offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,G2,Worst,4U>(equities,rough,
        {{{0.6f,0.04f,0.9f,0.03f,0.3f},{0.02f,0.01f}}},
        {0U},1U,{{0.0f,8U}},{matrix},external,internal,
        {{0U,0U,0U,0U,1.0f}},128U,128U,
        1.0f/252.0f,1U,32U,2041U);
    require_close(team4.prices[0],result.prices[0],2.0e-6f,
        "rough G2 price depends on TeamSize");
    require_close(team4.standard_errors[0],
        result.standard_errors[0],2.0e-6f,
        "rough G2 error depends on TeamSize");

}
void batched_geometry_parity() {
    using OU=model::fixed_income::ornstein_uhlenbeck::CorrelatedRatePolicy;
    offline::pricing::MultiAssetHostModels<Pack> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f},
                     {1.1f,0.0f,0.0f,0.3f}};
    equities.scenario_indices={0U,1U,1U,0U};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> names{"a.spot","b.spot","rate.x"};
    const std::vector<correlation::MatrixRow> matrices{
        {"identity",names,correlation::packed_cholesky({
            1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0},3U)},
        {"correlated",names,correlation::packed_cholesky({
            1.0,0.25,0.1,0.25,1.0,-0.1,0.1,-0.1,1.0},3U)}};
    const std::vector<Worst::ProductParameters> products{
        {0.85f,7U},{0.95f,11U},{1.05f,15U}};
    std::vector<monte_carlo::MultiAssetPriceRow> rows;
    for(unsigned i=0U;i<19U;++i)
        rows.push_back({i%2U,i%3U,(i/3U)%2U,0U,1.0f});
    const auto run=[&]<unsigned TeamSize>(
        offline::pricing::CrossAssetBatchGeometry geometry) {
        return offline::pricing::execute_cross_asset_prices<
            Pack,NoRough,OU,Worst,TeamSize>(
                equities,rough,{{{0.6f,0.04f},0.02f},
                                 {{0.9f,0.07f},0.03f}},
                {0U,1U},2U,products,matrices,names,names,rows,
                513U,513U,1.0f/252.0f,1U,32U,17521U,{},geometry);
    };
    const auto reference=run.template operator()<1U>({19U,1U,128U});
    const auto multi=run.template operator()<1U>({19U,4U,128U});
    const auto wide=run.template operator()<1U>({19U,2U,256U});
    const auto teams=run.template operator()<4U>({19U,2U,128U});
    for(std::size_t i=0U;i<rows.size();++i) {
        for(const auto* candidate:{&multi,&wide,&teams}) {
            require_close(candidate->prices[i],reference.prices[i],
                2.0e-6f,"batched price depends on grid geometry");
            require_close(candidate->standard_errors[i],
                reference.standard_errors[i],2.0e-6f,
                "batched error depends on grid geometry");
        }
    }
}
}
int main() {
    all_rate_families();
    constant_rate_team();
    g2_internal_rho_rejected();
    g2_gaussian_integral();
    g2_with_rough();
    benchmark_rate_families();
    large_shared_matrix();
    batched_geometry_parity();
    std::cout<<"all cross-asset rate families passed\n";
}
