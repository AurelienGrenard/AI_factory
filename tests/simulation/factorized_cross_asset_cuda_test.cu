// Factor-only model rows, global Brownian correlation, and zero-factor rate.
#include "common/simulation/rate_adjusted_equity_path.cuh"
#include "correlation/dataset.hpp"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "model/fixed_income/constant_rate/correlated_dynamics.cuh"
#include "model/fixed_income/g2/correlated_dynamics.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/cross_asset_price_generation.cuh"
#include "tools/pricing/factorized_scenario_assembly.cuh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    using Equity = model::equity::heston::DynamicsPolicy;
    using Pack = simulation::MultiAssetPack<
        simulation::RateAdjustedModelGroup<Equity,1U>>;
    using Rough = simulation::MultiRoughPack<>;
    using Rate = model::fixed_income::constant_rate::CorrelatedRatePolicy;
    using Product = product::WorstOfCallPathPolicy;

    offline::pricing::MultiAssetHostModels<Pack> factor_equity;
    factor_equity.models = {{
        1.0f, 0.0f, 0.01f, 0.04f, 1.5f, 0.04f, 0.3f, 0.0f}};
    factor_equity.scenario_indices = {0U};
    offline::pricing::MultiRoughHostModels<Rough> rough;
    const std::vector<Rate::Parameters> rates{{0.03f}};
    const std::vector<std::string> names{
        "asset.spot", "asset.variance"};
    const correlation::MatrixRow matrix{
        "heston", names,
        correlation::packed_cholesky({
            1.0, -0.6,
            -0.6, 1.0
        }, 2U)};
    const std::vector<monte_carlo::MultiAssetPriceRow> rows{
        {0U,0U,0U,0U,1.0f}};
    const auto bound = offline::pricing::assemble_factorized_scenarios<
        Pack,Rough,Rate>(
            factor_equity,rough,rates,{0U},{0U},1U,
            {matrix},names,names,rows);
    if (bound.scenario_count != 1U
        || bound.equities.models.size() != 1U
        || std::abs(bound.equities.models[0].rho + 0.6f) > 1.0e-6f
        || bound.equities.models[0].risk_free_rate != 0.0f)
        throw std::runtime_error("Factorized Heston binding failed");

    const correlation::MatrixRow alternative{
        "other", names, correlation::packed_cholesky({
            1.0, 0.2,
            0.2, 1.0
        }, 2U)};
    const auto variants = offline::pricing::assemble_factorized_scenarios<
        Pack, Rough, Rate>(factor_equity, rough, {{0.01f}, {0.03f}},
            {1U}, {0U}, 1U,
            {matrix, alternative}, names, names,
            {{0U, 0U, 0U, 0U, 1.0f},
             {0U, 0U, 1U, 0U, 1.0f},
             {0U, 0U, 0U, 0U, 1.0f}});
    if (variants.scenario_count != 2U
        || variants.rows[0].scenario_index != 0U
        || variants.rows[1].scenario_index != 1U
        || variants.rows[2].scenario_index != 0U
        || variants.rate_indices != std::vector<std::uint32_t>{0U, 1U}
        || variants.source_rate_indices != std::vector<std::uint32_t>{1U, 1U}
        || std::abs(variants.equities.models[1].rho - 0.2f) > 1.0e-6f)
        throw std::runtime_error("Correlation scenario caching failed");

    using G2 = model::fixed_income::g2::CorrelatedRatePolicy;
    const std::vector<std::string> g2_names{
        "asset.spot", "asset.variance", "rate.x", "rate.y"};
    const correlation::MatrixRow g2_matrix{
        "g2", g2_names, correlation::packed_cholesky({
            1.0, -0.6, 0.0, 0.0,
            -0.6, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.35,
            0.0, 0.0, 0.35, 1.0
        }, 4U)};
    const std::vector<G2::Parameters> g2_rates{{
        {0.6f, 0.04f, 0.9f, 0.03f, 0.0f}, {0.02f, 0.01f}}};
    const auto g2_bound = offline::pricing::assemble_factorized_scenarios<
        Pack, Rough, G2>(factor_equity, rough, g2_rates, {0U}, {0U}, 1U,
            {g2_matrix}, g2_names, g2_names, rows);
    if (std::abs(g2_bound.rates[0].process.correlation - 0.35f) > 1.0e-6f)
        throw std::runtime_error("Factorized G2 binding failed");

    constexpr std::uint32_t steps_per_day = 2U;
    constexpr float dt = 1.0f / (252.0f * steps_per_day);
    const std::vector<Product::ProductParameters> products{
        {0.9f, 32U}};
    const auto actual = offline::pricing::execute_cross_asset_prices<
        Pack,Rough,Rate,Product>(
            bound.equities,bound.rough,bound.rates,bound.rate_indices,
            bound.scenario_count,products,{matrix},names,names,bound.rows,
            2048U,2048U,dt,steps_per_day,32U,519U);

    auto complete = factor_equity;
    complete.models[0].rho = -0.6f;
    const auto reference = offline::pricing::execute_cross_asset_prices<
        Pack,Rough,Rate,Product>(
            complete,rough,rates,{0U},1U,products,
            {matrix},names,names,rows,
            2048U,2048U,dt,steps_per_day,32U,519U);
    if (actual.prices.size() != 1U
        || !std::isfinite(actual.prices[0])
        || actual.prices[0] != reference.prices[0]
        || actual.standard_errors[0] != reference.standard_errors[0])
        throw std::runtime_error("Factorized Heston changed the price path");

    // The same binding must also cover a rough volatility-first first pass.
    using Bs = model::equity::black_scholes::DynamicsPolicy;
    using MixedEquity = simulation::MultiAssetPack<
        simulation::RateAdjustedModelGroup<Bs, 1U>>;
    using MixedRough = simulation::MultiRoughPack<
        model::equity::rough_heston::CausalFftPathPolicy>;
    offline::pricing::MultiAssetHostModels<MixedEquity> mixed_equity;
    mixed_equity.models = {{1.0f, 0.0f, 0.0f, 0.15f}};
    mixed_equity.scenario_indices = {0U};
    offline::pricing::MultiRoughHostModels<MixedRough> mixed_rough;
    mixed_rough.models = {{
        1.0f, 0.0f, 0.0f, 0.04f, 1.0f, 0.04f, 0.3f, 0.12f, 0.0f}};
    mixed_rough.scenario_indices = {0U};
    const std::vector<std::string> external{
        "bs.spot", "rough.spot", "rough.vol"};
    const std::vector<std::string> internal{
        "rough.vol", "bs.spot", "rough.spot"};
    const correlation::MatrixRow mixed_matrix{
        "mixed", external, correlation::packed_cholesky({
            1.0, 0.2, 0.0,
            0.2, 1.0, -0.4,
            0.0, -0.4, 1.0
        }, 3U)};
    const auto mixed_bound = offline::pricing::assemble_factorized_scenarios<
        MixedEquity, MixedRough, Rate>(
            mixed_equity, mixed_rough, {{0.02f}}, {0U}, {0U, 0U}, 1U,
            {mixed_matrix}, external, internal, rows);
    if (std::abs(mixed_bound.rough.models[0].rho + 0.4f) > 2.0e-5f
        || mixed_bound.rough.models[0].risk_free_rate != 0.0f)
        throw std::runtime_error("Factorized rough binding failed");
    const std::vector<Product::ProductParameters> mixed_products{
        {0.95f, 8U}};
    const auto mixed_actual = offline::pricing::execute_cross_asset_prices<
        MixedEquity, MixedRough, Rate, Product>(
            mixed_bound.equities, mixed_bound.rough,
            mixed_bound.rates, mixed_bound.rate_indices,
            mixed_bound.scenario_count, mixed_products, {mixed_matrix},
            external, internal, mixed_bound.rows,
            128U, 128U, dt, steps_per_day, 32U, 775U);
    mixed_rough.models[0].rho = mixed_bound.rough.models[0].rho;
    const auto mixed_reference = offline::pricing::execute_cross_asset_prices<
        MixedEquity, MixedRough, Rate, Product>(
            mixed_equity, mixed_rough, {{0.02f}}, {0U}, 1U,
            mixed_products, {mixed_matrix}, external, internal, rows,
            128U, 128U, dt, steps_per_day, 32U, 775U);
    if (mixed_actual.prices[0] != mixed_reference.prices[0]
        || mixed_actual.standard_errors[0] != mixed_reference.standard_errors[0])
        throw std::runtime_error("Factorized rough path changed the price");
}
