#include "correlation/dataset.hpp"
#include "common/simulation/rate_adjusted_equity_path.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "model/fixed_income/vasicek/correlated_dynamics.cuh"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "product/best_of_call/pricing_policy.cuh"
#include "product/basket_call/pricing_policy.cuh"
#include "tools/pricing/cross_asset_price_generation.cuh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace wb=ai_factory::workbench;
using BS=wb::model::equity::black_scholes::DynamicsPolicy;
using Rate=wb::model::fixed_income::vasicek::CorrelatedRatePolicy;
using Rough=wb::model::equity::rough_heston::CausalFftPathPolicy;
using Worst=wb::product::WorstOfCallPathPolicy;
using Best=wb::product::BestOfCallPathPolicy;
using Basket=wb::product::BasketCallPathPolicy;
using Barrier=wb::product::BasketDownAndOutCallPathPolicy;
using Expression=wb::RowIndexExpression;
using TwoBS=wb::simulation::MultiAssetPack<
    wb::simulation::RateAdjustedModelGroup<BS,2U>>;
using OneBS=wb::simulation::MultiAssetPack<
    wb::simulation::RateAdjustedModelGroup<BS,1U>>;
using NoRough=wb::simulation::MultiRoughPack<>;
using OneRough=wb::simulation::MultiRoughPack<Rough>;

void require(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(message);
}

wb::CompiledRowIndexPlan single_plan(unsigned slots) {
    std::vector<Expression> sources;
    for(unsigned i=0U;i<slots;++i)
        sources.push_back(Expression::source(i,1U));
    return wb::compile_row_index_plan(Expression::aligned(sources),slots);
}

void markov_rate() {
    wb::offline::pricing::MultiAssetHostModels<TwoBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    wb::offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<Rate::Parameters> rates{{{0.7f,0.025f,0.035f},0.02f}};
    const std::vector<std::string> factors{
        "a.spot","b.spot","rate.brownian"};
    const std::vector<wb::correlation::MatrixRow> matrices{{
        "correlated",factors,wb::correlation::packed_cholesky({
            1.0,0.2,0.4,0.2,1.0,-0.3,0.4,-0.3,1.0},3U)}};
    const auto plan=single_plan(5U);
    const auto martingale=wb::offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Worst>(equities,rough,rates,
            {{0.0f,32U}},matrices,factors,factors,plan,
            2048U,2048U,1.0f/252.0f,1U,32U,47511U);
    require(std::abs(martingale.prices[0]-1.0f)<4e-5f,
        "discounted zero-volatility basket is not a martingale");
    equities.models[0].volatility=0.25f;
    const auto one=wb::offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Barrier,1U>(equities,rough,rates,
            {{0.8f,0.72f,32U,4U}},matrices,factors,factors,plan,
            1024U,1024U,1.0f/252.0f,1U,32U,4511U);
    const auto four=wb::offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Barrier,4U>(equities,rough,rates,
            {{0.8f,0.72f,32U,4U}},matrices,factors,factors,plan,
            1024U,1024U,1.0f/252.0f,1U,32U,4511U);
    require(std::abs(one.prices[0]-four.prices[0])<2e-6f
        &&std::abs(one.standard_errors[0]-four.standard_errors[0])<2e-6f,
        "rate basket depends on path team size");
}

void rough_rate() {
    wb::offline::pricing::MultiAssetHostModels<OneBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f}};
    wb::offline::pricing::MultiRoughHostModels<OneRough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.04f,2.0f,0.08f,
                   0.3f,0.12f,-0.4f}};
    const std::vector<Rate::Parameters> rates{{{0.7f,0.025f,0.035f},0.02f}};
    const std::vector<std::string> external{
        "bs.spot","rough.spot","rough.vol","rate.brownian"};
    const std::vector<std::string> internal{
        "rough.vol","bs.spot","rate.brownian","rough.spot"};
    const wb::correlation::MatrixRow raw{
        "rough_rate",external,wb::correlation::packed_cholesky({
            1.0,0.0,0.0,0.2,
            0.0,1.0,-0.4,0.0,
            0.0,-0.4,1.0,0.0,
            0.2,0.0,0.0,1.0},4U)};
    const auto plan=single_plan(5U);
    const auto full=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Barrier,1U>(equities,rough,rates,
            {{0.85f,0.75f,32U,4U}},{raw},external,internal,plan,
            512U,512U,1.0f/252.0f,1U,32U,4511U);
    const auto chunked=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Barrier,4U>(equities,rough,rates,
            {{0.85f,0.75f,32U,4U}},{raw},external,internal,plan,
            512U,128U,1.0f/252.0f,1U,32U,4511U);
    require(std::isfinite(full.prices[0]) && full.prices[0]>=0.0f,
        "rough and rate basket returned invalid price");
    require(std::abs(full.prices[0]-chunked.prices[0])<2e-6f
        &&std::abs(full.standard_errors[0]-chunked.standard_errors[0])<2e-6f,
        "rough and rate basket depends on path chunk or team size");

    const auto worst=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Worst,4U>(equities,rough,rates,
            {{1.0f,32U}},{raw},external,internal,plan,
            512U,128U,1.0f/252.0f,1U,32U,4511U);
    const auto basket=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Basket,4U>(equities,rough,rates,
            {{1.0f,32U}},{raw},external,internal,plan,
            512U,128U,1.0f/252.0f,1U,32U,4511U);
    const auto best=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Best,4U>(equities,rough,rates,
            {{1.0f,32U}},{raw},external,internal,plan,
            512U,128U,1.0f/252.0f,1U,32U,4511U);
    require(std::isfinite(worst.prices[0]) && worst.prices[0]>=0.0f
        &&worst.prices[0]<=basket.prices[0]
        &&basket.prices[0]<=best.prices[0],
        "rough/rate terminal basket payoffs are misordered");
}

int main() {
    markov_rate();
    rough_rate();
}
