// Correlated stochastic-rate discounting for equity baskets, with and without rough volatility.
#include "correlation/dataset.hpp"
#include "common/simulation/rate_adjusted_equity_path.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/markovian/cev/dynamics_impl.cuh"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/hybrid_multi_asset.cuh"
#include "model/fixed_income/vasicek/correlated_dynamics.cuh"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/cross_asset_price_generation.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace ai_factory::workbench;
using BS=model::equity::black_scholes::DynamicsPolicy;
using Rate=model::fixed_income::vasicek::CorrelatedRatePolicy;
using Heston=model::equity::rough_heston::CausalFftPathPolicy;
using TwoBS=simulation::MultiAssetPack<
    simulation::RateAdjustedModelGroup<BS,2U>>;
using OneBS=simulation::MultiAssetPack<
    simulation::RateAdjustedModelGroup<BS,1U>>;
using NoRough=simulation::MultiRoughPack<>;
using OneRough=simulation::MultiRoughPack<Heston>;
using Bergomi=model::equity::rough_multi_asset::HybridPolicy<
    model::equity::rough_bergomi::PathPolicy>;
using OneHybridRough=simulation::MultiRoughPack<Bergomi>;
using Worst=product::WorstOfCallPathPolicy;
using Barrier=product::BasketDownAndOutCallPathPolicy;

void require(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(message);
}

void pure_basket() {
    offline::pricing::MultiAssetHostModels<TwoBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    equities.scenario_indices={0U,0U};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<Rate::Parameters> rates{
        {{0.7f,0.025f,0.035f},0.02f}};
    const std::vector<std::string> names{
        "a.spot","b.spot","rate.brownian"};
    const correlation::MatrixRow matrix{
        "correlated",names,correlation::packed_cholesky({
            1.0,0.2,0.4,
            0.2,1.0,-0.3,
            0.4,-0.3,1.0
        },3U)};
    const std::vector<monte_carlo::MultiAssetPriceRow> rows{
        {0U,0U,0U,0U,1.0f}};
    const auto run=offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Worst>(
            equities,rough,rates,{0U},1U,{{0.0f,32U}},
            {matrix},names,names,rows,4096U,4096U,
            1.0f/252.0f,1U,32U,47511U);
    require(std::abs(run.prices[0]-1.0f)<3.0e-5f,
        "discounted zero-volatility equity basket is not a martingale");
    require(run.standard_errors[0]<1.0e-5f,
        "zero-volatility basket payoff has unexpected dispersion");
    auto zero_reversion=rates;
    zero_reversion[0].process.mean_reversion=0.0f;
    const auto brownian_rate=offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Worst>(
            equities,rough,zero_reversion,{0U},1U,{{0.0f,32U}},
            {matrix},names,names,rows,1024U,1024U,
            1.0f/252.0f,1U,32U,47511U);
    require(std::abs(brownian_rate.prices[0]-1.0f)<3.0e-5f,
        "zero mean-reversion rate branch incorrect");
    // With zero equity volatility, S_T=exp(I_T) and the option has a
    // closed Gaussian integral-rate price.
    auto annual_rate=rates;
    annual_rate[0].process.volatility=0.1f;
    const auto annual=offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Worst>(
            equities,rough,annual_rate,{0U},1U,{{1.0f,252U}},
            {matrix},names,names,rows,8192U,8192U,
            1.0f/252.0f,1U,32U,15317U);
    const double a=0.7,b=0.025,r0=0.02,sigma=0.1;
    const double decay=std::exp(-a);
    const double mean=b+(r0-b)*(1.0-decay)/a;
    const double variance=sigma*sigma/(a*a)
        *(1.0-2.0*(1.0-decay)/a+(1.0-decay*decay)/(2.0*a));
    const double sd=std::sqrt(variance);
    const auto normal_cdf=[](double z) {
        return 0.5*std::erfc(-z/std::sqrt(2.0));
    };
    const double reference=normal_cdf(mean/sd)
        -std::exp(-mean+0.5*variance)
            *normal_cdf((mean-variance)/sd);
    require(std::abs(static_cast<double>(annual.prices[0])-reference)
        <5.0*annual.standard_errors[0]+2.0e-4,
        "integrated Vasicek Gaussian option price differs from closed form");
    bool rejected=false;
    try {
        auto invalid=rows;
        invalid[0].discount_factor=0.98f;
        (void)offline::pricing::execute_cross_asset_prices<
            TwoBS,NoRough,Rate,Worst>(
                equities,rough,rates,{0U},1U,{{0.0f,32U}},
                {matrix},names,names,invalid,128U,128U,
                1.0f/252.0f,1U,32U,47511U);
    } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"deterministic discount combined with simulated rate");
}


void team_size_invariance() {
    offline::pricing::MultiAssetHostModels<TwoBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.25f}};
    equities.scenario_indices={0U,0U};
    offline::pricing::MultiRoughHostModels<NoRough> no_rough;
    const std::vector<Rate::Parameters> rates{
        {{0.7f,0.025f,0.035f},0.02f}};
    const std::vector<std::string> names{
        "a.spot","b.spot","rate.brownian"};
    const correlation::MatrixRow matrix{
        "team",names,correlation::packed_cholesky({
            1.0,0.2,0.4,
            0.2,1.0,-0.3,
            0.4,-0.3,1.0},3U)};
    const std::vector<monte_carlo::MultiAssetPriceRow> rows{
        {0U,0U,0U,0U,1.0f}};
    const auto markov=[&]<unsigned TeamSize>() {
        return offline::pricing::execute_cross_asset_prices<
            TwoBS,NoRough,Rate,Barrier,TeamSize>(
                equities,no_rough,rates,{0U},1U,
                {{0.8f,0.72f,32U,4U}},{matrix},names,names,
                rows,1024U,1024U,1.0f/252.0f,1U,32U,4511U);
    };
    const auto reference=markov.template operator()<1U>();
    const auto two=markov.template operator()<2U>();
    const auto four=markov.template operator()<4U>();
    const auto eight=markov.template operator()<8U>();
    for(const auto* run:{&two,&four,&eight}) {
        require(std::abs(run->prices[0]-reference.prices[0])<2.0e-6f
            && std::abs(run->standard_errors[0]
                -reference.standard_errors[0])<2.0e-6f,
            "markov/rate result depends on TeamSize");
    }

    offline::pricing::MultiAssetHostModels<OneBS> one_equity;
    one_equity.models={{1.0f,0.0f,0.0f,0.2f}};
    one_equity.scenario_indices={0U};
    offline::pricing::MultiRoughHostModels<OneRough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.04f,2.0f,0.08f,
                   0.3f,0.12f,-0.4f}};
    rough.scenario_indices={0U};
    const std::vector<std::string> external{
        "bs.spot","rough.spot","rough.vol","rate.brownian"};
    const std::vector<std::string> internal{
        "rough.vol","bs.spot","rate.brownian","rough.spot"};
    const correlation::MatrixRow internal_matrix{
        "rough_team",internal,correlation::packed_cholesky({
            1.0,0.0,0.2,-0.4,
            0.0,1.0,0.25,0.0,
            0.2,0.25,1.0,0.15,
            -0.4,0.0,0.15,1.0},4U)};
    const auto rough_matrix=correlation::reorder_factors(
        internal_matrix,external);
    const auto mixed=[&]<unsigned TeamSize>() {
        return offline::pricing::execute_cross_asset_prices<
            OneBS,OneRough,Rate,Barrier,TeamSize>(
                one_equity,rough,rates,{0U},1U,
                {{0.85f,0.75f,32U,4U}},
                {rough_matrix},external,internal,rows,
                512U,128U,1.0f/252.0f,1U,32U,4511U);
    };
    const auto rough_reference=mixed.template operator()<1U>();
    const auto rough_two=mixed.template operator()<2U>();
    const auto rough_four=mixed.template operator()<4U>();
    for(const auto* run:{&rough_two,&rough_four}) {
        require(std::abs(run->prices[0]-rough_reference.prices[0])<2.0e-6f
            && std::abs(run->standard_errors[0]
                -rough_reference.standard_errors[0])<2.0e-6f,
            "rough/rate result depends on TeamSize");
    }
}


void pathwise_team_size_invariance() {
    constexpr std::size_t paths=128U;
    constexpr std::uint32_t steps=16U;
    offline::pricing::MultiAssetHostModels<OneBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f}};
    equities.scenario_indices={0U};
    offline::pricing::MultiRoughHostModels<OneRough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.04f,2.0f,0.08f,
                   0.3f,0.12f,-0.4f}};
    rough.scenario_indices={0U};
    offline::pricing::MultiAssetDeviceModels<OneBS> device_equities(
        equities);
    offline::pricing::MultiRoughDeviceModels<OneRough> device_rough(
        rough);
    const Rate::Parameters rate{{0.6f,0.02f,0.03f},0.02f};
    const Worst::ProductParameters product{0.8f,steps};
    const monte_carlo::MultiAssetPriceRow row{0U,0U,0U,0U,1.0f};
    const std::vector<std::string> names{
        "rough.vol","bs.spot","rate.brownian","rough.spot"};
    const correlation::MatrixRow matrix{
        "pathwise",names,correlation::packed_cholesky({
            1.0,0.1,0.2,-0.4,
            0.1,1.0,0.25,0.0,
            0.2,0.25,1.0,0.1,
            -0.4,0.0,0.1,1.0},4U)};
    offline::cuda::DeviceBuffer<Rate::Parameters> device_rate(1U);
    offline::cuda::DeviceBuffer<std::uint32_t> device_rate_index(1U);
    offline::cuda::DeviceBuffer<Worst::ProductParameters>
        device_product(1U);
    offline::cuda::DeviceBuffer<monte_carlo::MultiAssetPriceRow>
        device_row(1U);
    offline::cuda::DeviceBuffer<float> device_factors(
        matrix.lower.size());
    offline::cuda::DeviceBuffer<float> device_volatility(paths*steps);
    offline::cuda::DeviceBuffer<volterra::PartialMoments>
        device_moments(1U);
    offline::cuda::DeviceBuffer<float> device_payoffs(paths);
    const std::uint32_t rate_index=0U;
    std::vector<float> constant_volatility(paths*steps,0.04f);
    device_rate.copy_from(&rate);
    device_rate_index.copy_from(&rate_index);
    device_product.copy_from(&product);
    device_row.copy_from(&row);
    device_factors.copy_from(matrix.lower.data());
    device_volatility.copy_from(constant_volatility.data());
    const monte_carlo::CrossAssetDeviceInputs<
        OneBS,OneRough,Rate,Worst> inputs{
            device_equities.inputs(),device_rough.inputs(),
            device_rate.data(),device_rate_index.data(),nullptr,
            device_product.data(),device_row.data(),
            device_factors.data()};
    const auto run=[&]<unsigned TeamSize>() {
        monte_carlo::launch_cross_asset_cuda<
            OneBS,OneRough,Rate,Worst,true,TeamSize>(
                inputs,0U,0U,paths,steps,1U,1.0f/252.0f,
                19431U,device_volatility.data(),
                device_moments.data(),device_payoffs.data());
        std::vector<float> values(paths);
        device_payoffs.copy_to(values.data());
        return values;
    };
    const auto reference=run.template operator()<1U>();
    require(reference!=std::vector<float>(paths,0.0f),
        "pathwise reference payoff is trivial");
    require(run.template operator()<2U>()==reference
        &&run.template operator()<4U>()==reference
        &&run.template operator()<8U>()==reference,
        "joint rough/rate path payoffs depend on TeamSize");
}

void benchmark_large_basket() {
    using Plain=simulation::MultiAssetPack<
        simulation::ModelGroup<BS,30U>>;
    using Adjusted=simulation::MultiAssetPack<
        simulation::RateAdjustedModelGroup<BS,30U>>;
    offline::pricing::MultiAssetHostModels<Plain> plain;
    offline::pricing::MultiAssetHostModels<Adjusted> adjusted;
    plain.models={{1.0f,0.0f,0.0f,0.2f}};
    adjusted.models=plain.models;
    plain.scenario_indices.assign(30U,0U);
    adjusted.scenario_indices=plain.scenario_indices;
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    std::vector<std::string> plain_names,adjusted_names;
    std::vector<double> plain_identity(30U*30U,0.0);
    std::vector<double> adjusted_identity(31U*31U,0.0);
    for(std::size_t i=0U;i<30U;++i) {
        plain_names.push_back("asset_"+std::to_string(i)+".spot");
        plain_identity[i*30U+i]=1.0;
        adjusted_identity[i*31U+i]=1.0;
    }
    adjusted_names=plain_names;
    adjusted_names.push_back("rate.brownian");
    adjusted_identity[30U*31U+30U]=1.0;
    const correlation::MatrixRow plain_matrix{
        "plain",plain_names,correlation::packed_cholesky(
            plain_identity,30U)};
    const correlation::MatrixRow adjusted_matrix{
        "adjusted",adjusted_names,correlation::packed_cholesky(
            adjusted_identity,31U)};
    const std::vector<monte_carlo::MultiAssetPriceRow> rows{
        {0U,0U,0U,0U,1.0f}};
    offline::pricing::BatchedMonteCarloProfile profile{};
    profile.paths_per_price=8192U;
    profile.results_per_launch=1U;
    profile.block_count_limit=1U;
    profile.threads_per_block=128U;
    profile.seed=4343U;
    std::vector<double> plain_times,adjusted_times;
    std::vector<double> team2_times,team4_times,team8_times;
    for(unsigned trial=0U;trial<5U;++trial) {
        const auto p=offline::pricing::execute_multi_asset_prices<
            Plain,Worst,1U>(
                plain,1U,{{0.7f,21U}},{plain_matrix},
                plain_names,rows,profile,1.0f/252.0f,1U);
        const auto q=offline::pricing::execute_cross_asset_prices<
            Adjusted,NoRough,Rate,Worst>(
                adjusted,rough,{{{0.6f,0.02f,0.05f},0.02f}},
                {0U},1U,{{0.7f,21U}},{adjusted_matrix},
                adjusted_names,adjusted_names,rows,8192U,8192U,
                1.0f/252.0f,1U,32U,4343U);
        const auto q2=offline::pricing::execute_cross_asset_prices<
            Adjusted,NoRough,Rate,Worst,2U>(
                adjusted,rough,{{{0.6f,0.02f,0.05f},0.02f}},
                {0U},1U,{{0.7f,21U}},{adjusted_matrix},
                adjusted_names,adjusted_names,rows,8192U,8192U,
                1.0f/252.0f,1U,32U,4343U);
        const auto q4=offline::pricing::execute_cross_asset_prices<
            Adjusted,NoRough,Rate,Worst,4U>(
                adjusted,rough,{{{0.6f,0.02f,0.05f},0.02f}},
                {0U},1U,{{0.7f,21U}},{adjusted_matrix},
                adjusted_names,adjusted_names,rows,8192U,8192U,
                1.0f/252.0f,1U,32U,4343U);
        const auto q8=offline::pricing::execute_cross_asset_prices<
            Adjusted,NoRough,Rate,Worst,8U>(
                adjusted,rough,{{{0.6f,0.02f,0.05f},0.02f}},
                {0U},1U,{{0.7f,21U}},{adjusted_matrix},
                adjusted_names,adjusted_names,rows,8192U,8192U,
                1.0f/252.0f,1U,32U,4343U);
        plain_times.push_back(p.kernel_seconds);
        adjusted_times.push_back(q.kernel_seconds);
        team2_times.push_back(q2.kernel_seconds);
        team4_times.push_back(q4.kernel_seconds);
        team8_times.push_back(q8.kernel_seconds);
        for(const auto* run:{&q2,&q4,&q8})
            require(std::abs(run->prices[0]-q.prices[0])<2.0e-6f
                && std::abs(run->standard_errors[0]
                    -q.standard_errors[0])<2.0e-6f,
                "large basket result depends on TeamSize");
        require(std::isfinite(q.prices[0]),
            "large basket with stochastic rate is invalid");
    }
    std::sort(plain_times.begin(),plain_times.end());
    std::sort(adjusted_times.begin(),adjusted_times.end());
    std::sort(team2_times.begin(),team2_times.end());
    std::sort(team4_times.begin(),team4_times.end());
    std::sort(team8_times.begin(),team8_times.end());
    std::cout<<"D=30+rate baseline_ms="<<plain_times[2U]*1000.0
        <<" team1_ms="<<adjusted_times[2U]*1000.0
        <<" team2_ms="<<team2_times[2U]*1000.0
        <<" team4_ms="<<team4_times[2U]*1000.0
        <<" team8_ms="<<team8_times[2U]*1000.0<<'\n';
}

void correlated_rate_spot() {
    offline::pricing::MultiAssetHostModels<OneBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f}};
    equities.scenario_indices={0U};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> names{"spot","rate.brownian"};
    constexpr double rho=0.65;
    const correlation::MatrixRow matrix{
        "spot_rate",names,correlation::packed_cholesky({
            1.0,rho,rho,1.0
        },2U)};
    const auto result=offline::pricing::execute_cross_asset_prices<
        OneBS,NoRough,Rate,Worst>(
            equities,rough,{{{0.7f,0.025f,0.1f},0.02f}},
            {0U},1U,{{1.0f,252U}},{matrix},names,names,
            {{0U,0U,0U,0U,1.0f}},16384U,16384U,
            1.0f/252.0f,1U,32U,1541U);
    const double a=0.7,b=0.025,r0=0.02,sigma_r=0.1;
    const double sigma_s=0.2,decay=std::exp(-a);
    const double loading=(1.0-decay)/a;
    const double mean_i=b+(r0-b)*loading;
    const double var_i=sigma_r*sigma_r/(a*a)
        *(1.0-2.0*loading+(1.0-decay*decay)/(2.0*a));
    const double cov_i_w=rho*sigma_r/a*(1.0-loading);
    const double mean_x=mean_i-0.5*sigma_s*sigma_s;
    const double var_x=var_i+sigma_s*sigma_s
        +2.0*sigma_s*cov_i_w;
    const double cov_x_i=var_i+sigma_s*cov_i_w;
    const auto normal_cdf=[](double z) {
        return 0.5*std::erfc(-z/std::sqrt(2.0));
    };
    const double reference=normal_cdf(
        (mean_x+var_x-cov_x_i)/std::sqrt(var_x))
        -std::exp(-mean_i+0.5*var_i)*normal_cdf(
            (mean_x-cov_x_i)/std::sqrt(var_x));
    require(std::abs(static_cast<double>(result.prices[0])-reference)
        <5.0*result.standard_errors[0]+3.0e-4,
        "spot-rate correlation disagrees with joint Gaussian price");
}

void absorbing_schemes() {
    using CEV=model::equity::cev::DynamicsPolicy;
    using SABR=model::equity::sabr::DynamicsPolicy;
    using Pack=simulation::MultiAssetPack<
        simulation::RateAdjustedModelGroup<CEV,1U>,
        simulation::RateAdjustedModelGroup<SABR,1U>>;
    offline::pricing::MultiAssetHostModels<Pack> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f,0.8f}};
    equities.scenario_indices={0U};
    equities.tail.models={{1.0f,0.0f,0.0f,0.2f,0.4f,-0.3f,0.6f}};
    equities.tail.scenario_indices={0U};
    offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> names{
        "cev.spot","sabr.spot","sabr.vol","rate.brownian"};
    const correlation::MatrixRow matrix{
        "absorbing",names,correlation::packed_cholesky({
            1.0,0.0,0.0,0.25,
            0.0,1.0,-0.3,0.1,
            0.0,-0.3,1.0,0.0,
            0.25,0.1,0.0,1.0
        },4U)};
    const auto result=offline::pricing::execute_cross_asset_prices<
        Pack,NoRough,Rate,Worst>(
            equities,rough,{{{0.5f,0.02f,0.025f},0.02f}},
            {0U},1U,{{0.9f,16U}},{matrix},names,names,
            {{0U,0U,0U,0U,1.0f}},256U,256U,
            1.0f/252.0f,1U,32U,3381U);
    require(std::isfinite(result.prices[0])
        && result.prices[0]>=0.0f,
        "CEV/SABR cross-asset price invalid");
}

void rough_basket() {
    offline::pricing::MultiAssetHostModels<OneBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    equities.scenario_indices={0U};
    offline::pricing::MultiRoughHostModels<OneRough> rough;
    Heston::Parameters heston{
        1.0f,0.0f,0.0f,0.0f,2.0f,0.0f,0.0f,0.12f,-0.4f};
    rough.models={heston};
    rough.scenario_indices={0U};
    const std::vector<Rate::Parameters> rates{
        {{0.6f,0.025f,0.035f},0.02f}};
    const std::vector<std::string> external{
        "bs.spot","rough.spot","rough.vol","rate.brownian"};
    const std::vector<std::string> internal{
        "rough.vol","bs.spot","rate.brownian","rough.spot"};
    const correlation::MatrixRow internal_matrix{
        "mixed",internal,correlation::packed_cholesky({
            1.0,0.0,0.2,-0.4,
            0.0,1.0,0.25,0.0,
            0.2,0.25,1.0,0.15,
            -0.4,0.0,0.15,1.0
        },4U)};
    const auto matrix=correlation::reorder_factors(
        internal_matrix,external);
    const std::vector<monte_carlo::MultiAssetPriceRow> rows{
        {0U,0U,0U,0U,1.0f}};
    const auto run=[&](std::size_t chunk) {
        return offline::pricing::execute_cross_asset_prices<
            OneBS,OneRough,Rate,Worst>(
                equities,rough,rates,{0U},1U,{{0.0f,32U}},
                {matrix},external,internal,rows,512U,chunk,
                1.0f/252.0f,1U,32U,1029U);
    };
    const auto full=run(512U);
    const auto chunked=run(128U);
    require(std::abs(full.prices[0]-1.0f)<3.0e-5f,
        "discounted deterministic rough and Markov basket incorrect");
    require(full.prices==chunked.prices
        && std::abs(full.standard_errors[0]
            -chunked.standard_errors[0])<1.0e-8f,
        "cross-asset rough result depends on chunking");
    rough.models[0].initial_variance=0.04f;
    rough.models[0].variance_drift=0.08f;
    rough.models[0].mean_reversion=2.0f;
    rough.models[0].volatility_of_variance=0.3f;
    const auto stochastic=run(128U);
    const auto stochastic_full=run(512U);
    require(stochastic.prices==stochastic_full.prices
        && std::abs(stochastic.standard_errors[0]
            -stochastic_full.standard_errors[0])<1.0e-7f,
        "stochastic rough result depends on chunking");
    require(std::isfinite(stochastic.prices[0])
        && stochastic.prices[0]>=0.0f,
        "stochastic rough cross-asset payoff invalid");
    const auto barrier=offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Barrier>(
            equities,rough,rates,{0U},1U,{{0.9f,0.8f,32U,4U}},
            {matrix},external,internal,rows,512U,128U,
            1.0f/252.0f,1U,32U,1029U);
    require(std::isfinite(barrier.prices[0]),
        "cross-asset rough barrier payoff invalid");
}


void benchmark_large_rough_basket() {
    using Large=simulation::MultiAssetPack<
        simulation::RateAdjustedModelGroup<BS,30U>>;
    offline::pricing::MultiAssetHostModels<Large> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f}};
    equities.scenario_indices.assign(30U,0U);
    offline::pricing::MultiRoughHostModels<OneRough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.04f,2.0f,0.08f,
                   0.3f,0.12f,-0.4f}};
    rough.scenario_indices={0U};
    std::vector<std::string> external,internal;
    for(unsigned asset=0U;asset<30U;++asset)
        external.push_back("bs_"+std::to_string(asset)+".spot");
    external.push_back("rough.spot");
    external.push_back("rough.vol");
    external.push_back("rate.brownian");
    internal.push_back("rough.vol");
    for(unsigned asset=0U;asset<30U;++asset)
        internal.push_back("bs_"+std::to_string(asset)+".spot");
    internal.push_back("rate.brownian");
    internal.push_back("rough.spot");
    constexpr unsigned n=33U;
    std::vector<double> dense(n*n,0.0);
    for(unsigned i=0U;i<n;++i) dense[i*n+i]=1.0;
    dense[0U*n+32U]=dense[32U*n+0U]=-0.4;
    dense[1U*n+31U]=dense[31U*n+1U]=0.2;
    const correlation::MatrixRow ordered{
        "large_rough",internal,correlation::packed_cholesky(dense,n)};
    const auto matrix=correlation::reorder_factors(ordered,external);
    const std::vector<monte_carlo::MultiAssetPriceRow> rows{
        {0U,0U,0U,0U,1.0f}};
    constexpr std::size_t paths=2048U;
    auto measure=[&]<unsigned TeamSize>() {
        std::vector<double> milliseconds;
        float price=0.0f,error=0.0f;
        for(unsigned trial=0U;trial<3U;++trial) {
            const auto run=offline::pricing::execute_cross_asset_prices<
                Large,OneRough,Rate,Worst,TeamSize>(
                    equities,rough,{{{0.6f,0.02f,0.03f},0.02f}},
                    {0U},1U,{{0.7f,32U}},{matrix},external,internal,
                    rows,paths,paths,1.0f/252.0f,1U,32U,89111U);
            price=run.prices[0];
            error=run.standard_errors[0];
            milliseconds.push_back(run.kernel_seconds*1000.0);
        }
        std::sort(milliseconds.begin(),milliseconds.end());
        return std::pair{milliseconds[1U],std::pair{price,error}};
    };
    const auto team1=measure.template operator()<1U>();
    const auto team4=measure.template operator()<4U>();
    const auto team8=measure.template operator()<8U>();
    for(const auto* result:{&team4,&team8})
        require(std::abs(result->second.first-team1.second.first)<2.0e-6f
            &&std::abs(result->second.second-team1.second.second)
                <2.0e-6f,
            "large rough/rate result depends on TeamSize");
    std::cout<<"D=30+rough+rate paths="<<paths
        <<" team1_ms="<<team1.first
        <<" team4_ms="<<team4.first
        <<" team8_ms="<<team8.first<<'\n';
}

void rough_only_basket() {
    using NoMarkov=simulation::MultiAssetPack<>;
    offline::pricing::MultiAssetHostModels<NoMarkov> equities;
    offline::pricing::MultiRoughHostModels<OneRough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.0f,2.0f,0.0f,0.0f,0.12f,-0.4f}};
    rough.scenario_indices={0U};
    const std::vector<std::string> external{
        "rough.spot","rough.vol","rate.brownian"};
    const std::vector<std::string> internal{
        "rough.vol","rate.brownian","rough.spot"};
    const correlation::MatrixRow internal_matrix{
        "rough_only",internal,correlation::packed_cholesky({
            1.0,0.0,-0.4,
            0.0,1.0,0.2,
            -0.4,0.2,1.0
        },3U)};
    const auto matrix=correlation::reorder_factors(
        internal_matrix,external);
    const auto result=offline::pricing::execute_cross_asset_prices<
        NoMarkov,OneRough,Rate,Worst>(
            equities,rough,{{{0.5f,0.02f,0.03f},0.02f}},
            {0U},1U,{{0.0f,16U}},{matrix},external,internal,
            {{0U,0U,0U,0U,1.0f}},128U,128U,
            1.0f/252.0f,1U,32U,998U);
    require(std::abs(result.prices[0]-1.0f)<3.0e-5f,
        "discounted rough-only equity basket incorrect");
}

void hybrid_basket() {
    offline::pricing::MultiAssetHostModels<OneBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.12f}};
    equities.scenario_indices={0U};
    offline::pricing::MultiRoughHostModels<OneHybridRough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.04f,1.0f,0.12f,-0.4f}};
    rough.scenario_indices={0U};
    const std::vector<std::string> external{
        "bs.spot","bergomi.spot","bergomi.vol","rate.brownian"};
    const std::vector<std::string> internal{
        "bergomi.vol","bs.spot","rate.brownian","bergomi.spot"};
    const correlation::MatrixRow internal_matrix{
        "hybrid",internal,correlation::packed_cholesky({
            1.0,0.0,0.0,-0.4,
            0.0,1.0,0.2,0.0,
            0.0,0.2,1.0,0.1,
            -0.4,0.0,0.1,1.0
        },4U)};
    const auto matrix=correlation::reorder_factors(
        internal_matrix,external);
    const auto result=offline::pricing::execute_cross_asset_prices<
        OneBS,OneHybridRough,Rate,Worst>(
            equities,rough,{{{0.5f,0.02f,0.02f},0.02f}},
            {0U},1U,{{0.85f,16U}},{matrix},external,internal,
            {{0U,0U,0U,0U,1.0f}},128U,128U,
            1.0f/252.0f,1U,32U,989U);
    require(std::isfinite(result.prices[0])
        && result.prices[0]>=0.0f,
        "hybrid FFT cross-asset price invalid");
}
}

int main() {
    pure_basket();
    team_size_invariance();
    pathwise_team_size_invariance();
    correlated_rate_spot();
    absorbing_schemes();
    rough_basket();
    benchmark_large_rough_basket();
    rough_only_basket();
    hybrid_basket();
    benchmark_large_basket();
}
