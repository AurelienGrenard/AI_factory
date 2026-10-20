// One causal and one Gaussian-Volterra rough model in the same payoff.
#include "correlation/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/hybrid_multi_asset.cuh"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/heterogeneous_rough_price_generation.cuh"
#include "tools/pricing/mixed_rough_price_generation.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ai_factory::workbench;
using BS=model::equity::black_scholes::DynamicsPolicy;
using Markov=simulation::MultiAssetPack<simulation::ModelGroup<BS,1U>>;
using Heston=model::equity::rough_heston::CausalFftPathPolicy;
using Quadratic=model::equity::quadratic_rough_heston::CausalFftPathPolicy;
using Bergomi=model::equity::rough_multi_asset::HybridPolicy<
    model::equity::rough_bergomi::PathPolicy>;
using Heterogeneous=simulation::MultiRoughPack<Heston,Bergomi>;
using Homogeneous=simulation::MultiRoughPack<Heston,Heston>;
using Worst=product::WorstOfCallPathPolicy;
using Barrier=product::BasketDownAndOutCallPathPolicy;

void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
}

int main() {
    constexpr std::size_t paths=512U;
    constexpr std::uint32_t steps=64U;
    constexpr float dt=1.0f/252.0f;
    constexpr std::uint64_t seed=919133U;
    offline::pricing::MultiAssetHostModels<Markov> markov;
    markov.models={{1.0f,0.0f,0.0f,0.2f}};
    markov.scenario_indices={0U};
    const Heston::Parameters heston{
        1.0f,0.0f,0.0f,0.04f,2.0f,0.08f,0.3f,0.12f,-0.4f};
    const Bergomi::Parameters bergomi{
        1.0f,0.0f,0.0f,0.04f,1.0f,0.12f,-0.4f};
    offline::pricing::MultiRoughHostModels<Heterogeneous> rough;
    rough.models={heston};
    rough.scenario_indices={0U};
    rough.tail.models={bergomi};
    rough.tail.scenario_indices={0U};
    const std::vector<std::string> external{
        "bs.spot","heston.spot","heston.vol",
        "bergomi.spot","bergomi.vol"};
    const std::vector<std::string> internal{
        "heston.vol","bergomi.vol","bs.spot",
        "heston.spot","bergomi.spot"};
    const correlation::MatrixRow internal_matrix{
        "heterogeneous",internal,correlation::packed_cholesky({
            1.0,0.2,0.1,-0.4,0.0,
            0.2,1.0,0.0,0.0,-0.4,
            0.1,0.0,1.0,0.0,0.0,
            -0.4,0.0,0.0,1.0,0.0,
            0.0,-0.4,0.0,0.0,1.0
        },5U)};
    const auto matrix=correlation::reorder_factors(
        internal_matrix,external);
    const std::vector<monte_carlo::MultiAssetPriceRow> rows{
        {0U,0U,0U,0U,1.0f}};
    const std::vector<Worst::ProductParameters> products{{0.9f,steps}};
    const auto full=offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Heterogeneous,Worst>(
            markov,rough,1U,products,{matrix},external,internal,
            rows,paths,paths,dt,1U,32U,seed);
    const auto chunked=offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Heterogeneous,Worst>(
            markov,rough,1U,products,{matrix},external,internal,
            rows,paths,128U,dt,1U,32U,seed);
    require(std::isfinite(full.prices[0]) && full.prices[0]>=0.0f,
        "heterogeneous rough price invalid");
    require(full.prices==chunked.prices
        && full.standard_errors==chunked.standard_errors,
        "heterogeneous rough price depends on path chunking");
    const auto basket=offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Heterogeneous,Barrier>(
            markov,rough,1U,{{0.9f,0.8f,steps,4U}},
            {matrix},external,internal,rows,paths,paths,
            dt,1U,32U,seed);
    require(std::isfinite(basket.prices[0]),
        "heterogeneous rough basket barrier invalid");
    const auto basket_team2=
        offline::pricing::execute_heterogeneous_rough_prices<
            Markov,Heterogeneous,Barrier,2U>(
                markov,rough,1U,{{0.9f,0.8f,steps,4U}},
                {matrix},external,internal,rows,paths,paths,
                dt,1U,32U,seed);
    const auto basket_team4=
        offline::pricing::execute_heterogeneous_rough_prices<
            Markov,Heterogeneous,Barrier,4U>(
                markov,rough,1U,{{0.9f,0.8f,steps,4U}},
                {matrix},external,internal,rows,paths,paths,
                dt,1U,32U,seed);
    require(std::abs(basket_team2.prices[0]-basket.prices[0])<2.0e-6f
        && std::abs(basket_team4.prices[0]-basket.prices[0])<2.0e-6f
        && std::abs(basket_team2.standard_errors[0]
            -basket.standard_errors[0])<2.0e-6f
        && std::abs(basket_team4.standard_errors[0]
            -basket.standard_errors[0])<2.0e-6f,
        "heterogeneous rough barrier depends on TeamSize");

    // The recursive kernel agrees with the established homogeneous engine
    // when the two rough policies happen to be the same type.
    offline::pricing::MultiRoughHostModels<Homogeneous> same;
    same.models={heston};
    same.scenario_indices={0U};
    same.tail.models={heston};
    same.tail.scenario_indices={0U};
    const auto recursive=offline::pricing::execute_heterogeneous_rough_prices<
        Markov,Homogeneous,Worst>(
            markov,same,1U,products,{matrix},external,internal,
            rows,paths,paths,dt,1U,32U,seed);
    const auto established=offline::pricing::execute_mixed_rough_prices<
        Markov,Heston,Worst,2U>(
            markov,{heston,heston},{0U,1U},1U,products,{matrix},
            external,internal,rows,paths,paths,dt,1U,32U,seed);
    require(std::abs(recursive.prices[0]-established.prices[0])<2.0e-6f
        && std::abs(recursive.standard_errors[0]
            -established.standard_errors[0])<2.0e-6f,
        "recursive rough pack differs from homogeneous kernel");
    const auto recursive_team2=
        offline::pricing::execute_heterogeneous_rough_prices<
            Markov,Homogeneous,Worst,2U>(
                markov,same,1U,products,{matrix},external,internal,
                rows,paths,paths,dt,1U,32U,seed);
    const auto homogeneous_team2=
        offline::pricing::execute_mixed_rough_prices<
            Markov,Heston,Worst,2U,2U>(
                markov,{heston,heston},{0U,1U},1U,products,
                {matrix},external,internal,rows,paths,paths,
                dt,1U,32U,seed);
    require(std::abs(recursive_team2.prices[0]-recursive.prices[0])
            <2.0e-6f
        && std::abs(homogeneous_team2.prices[0]-established.prices[0])
            <2.0e-6f,
        "homogeneous rough result depends on TeamSize");
    std::vector<double> recursive_times,homogeneous_times;
    for(unsigned trial=0U;trial<7U;++trial) {
        recursive_times.push_back(
            offline::pricing::execute_heterogeneous_rough_prices<
                Markov,Homogeneous,Worst>(
                    markov,same,1U,products,{matrix},external,internal,
                    rows,paths,paths,dt,1U,32U,seed).kernel_seconds);
        homogeneous_times.push_back(
            offline::pricing::execute_mixed_rough_prices<
                Markov,Heston,Worst,2U>(
                    markov,{heston,heston},{0U,1U},1U,
                    products,{matrix},external,internal,rows,paths,
                    paths,dt,1U,32U,seed
                ).kernel_seconds);
    }
    std::sort(recursive_times.begin(),recursive_times.end());
    std::sort(homogeneous_times.begin(),homogeneous_times.end());
    std::cout<<"recursive_two_causal_ms="
             <<recursive_times[3U]*1000.0
             <<" homogeneous_two_causal_ms="
             <<homogeneous_times[3U]*1000.0<<'\n';

    // A one-factor rough model uses its volatility Brownian for its spot,
    // while the following two-factor model keeps a distinct spot factor.
    using MixedOneTwo=simulation::MultiRoughPack<Quadratic,Bergomi>;
    offline::pricing::MultiRoughHostModels<MixedOneTwo> one_two;
    one_two.models={{
        1.0f,0.0f,0.0f,0.05f,0.7f,0.1f,
        0.02f,1.4f,0.3f,0.12f}};
    one_two.scenario_indices={0U};
    one_two.tail.models={bergomi};
    one_two.tail.scenario_indices={0U};
    const std::vector<std::string> external_one_two{
        "bs.spot","quadratic.spot","bergomi.spot","bergomi.vol"};
    const std::vector<std::string> internal_one_two{
        "quadratic.spot","bergomi.vol","bs.spot","bergomi.spot"};
    const correlation::MatrixRow internal_one_two_matrix{
        "one_two",internal_one_two,correlation::packed_cholesky({
            1.0,0.1,0.0,0.0,
            0.1,1.0,0.0,-0.4,
            0.0,0.0,1.0,0.0,
            0.0,-0.4,0.0,1.0
        },4U)};
    const auto one_two_matrix=correlation::reorder_factors(
        internal_one_two_matrix,external_one_two);
    const auto one_two_price=
        offline::pricing::execute_heterogeneous_rough_prices<
            Markov,MixedOneTwo,Worst>(
                markov,one_two,1U,products,{one_two_matrix},
                external_one_two,internal_one_two,rows,paths,paths,
                dt,1U,32U,seed);
    require(std::isfinite(one_two_price.prices[0])
        && one_two_price.prices[0]>=0.0f,
        "one/two factor rough pack price invalid");
}
