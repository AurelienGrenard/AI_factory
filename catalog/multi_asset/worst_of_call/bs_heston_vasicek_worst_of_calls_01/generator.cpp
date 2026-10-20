// Two equity models and one stochastic numeraire share a single Brownian vector.
#include "common/simulation/rate_adjusted_equity_path.cuh"
#include "correlation/dataset.hpp"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/fixed_income/vasicek/correlated_dynamics.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"
#include "product/worst_of_call/dataset.hpp"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/datasets/multi_asset_price_dataset.hpp"
#include "tools/pricing/cross_asset_price_generation.cuh"
#include "tools/pricing/factorized_scenario_assembly.cuh"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    using Equity = simulation::MultiAssetPack<
        simulation::RateAdjustedModelGroup<
            model::equity::black_scholes::DynamicsPolicy, 1U>,
        simulation::RateAdjustedModelGroup<
            model::equity::heston::DynamicsPolicy, 1U>>;
    using Rough = simulation::MultiRoughPack<>;
    using Rate = model::fixed_income::vasicek::CorrelatedRatePolicy;
    using Product = product::WorstOfCallPathPolicy;
    constexpr std::uint32_t scenarios = 8U;
    constexpr std::size_t paths = 2048U;
    constexpr std::uint32_t steps_per_day = 2U;
    constexpr float dt = 1.0f / (252.0f * steps_per_day);
    constexpr std::uint64_t seed = 20261019ULL;
    const std::filesystem::path bs_path =
        "datasets/model/equity/markovian/black_scholes/factor_parameters/black_scholes_factors_01.json";
    const std::filesystem::path heston_path =
        "datasets/model/equity/markovian/heston/factor_parameters/heston_factors_01.json";
    const std::filesystem::path rate_path =
        "datasets/model/fixed_income/vasicek/parameters/vasicek_01.json";
    const std::filesystem::path product_path =
        "datasets/product/worst_of_call/worst_of_calls_01.json";
    const std::filesystem::path correlation_path =
        "datasets/correlation/bs_heston_vasicek_matrices_01.json";

    offline::pricing::MultiAssetHostModels<Equity> host;
    host.models = model::equity::black_scholes::load_models(bs_path);
    host.tail.models = model::equity::heston::load_models(heston_path);
    const auto rates = model::fixed_income::vasicek::load_models(rate_path);
    const auto products = product::load_worst_of_calls(product_path);
    const auto correlations = correlation::load_matrices(correlation_path);
    if (host.models.size() < scenarios || host.tail.models.size() < scenarios
        || rates.size() < scenarios || products.size() != 1U
        || correlations.size() != scenarios)
        throw std::invalid_argument("Incomplete factorized reference inputs");

    std::vector<std::uint32_t> source_model_indices;
    std::vector<std::uint32_t> rate_indices;
    std::vector<monte_carlo::MultiAssetPriceRow> rows;
    for (std::uint32_t i = 0U; i < scenarios; ++i) {
        host.scenario_indices.push_back(i);
        host.tail.scenario_indices.push_back(i);
        source_model_indices.insert(source_model_indices.end(), {i, i});
        rate_indices.push_back(i);
        rows.push_back({i, 0U, i, 0U, 1.0f});
    }
    const std::vector<std::string> factors{
        "asset_0.spot", "asset_1.spot", "asset_1.variance", "rate.factor"};
    const auto bound = offline::pricing::assemble_factorized_scenarios<
        Equity, Rough, Rate>(
            host, {}, rates, rate_indices, source_model_indices, scenarios,
            correlations, factors, factors, rows);
    const auto run = offline::pricing::execute_cross_asset_prices<
        Equity, Rough, Rate, Product>(
            bound.equities, bound.rough, bound.rates, bound.rate_indices,
            bound.scenario_count, products, correlations, factors, factors,
            bound.rows, paths, paths, dt, steps_per_day, 32U, seed);
    datasets::write_multi_asset_price_dataset(
        {{bs_path, heston_path}, product_path, correlation_path,
         std::nullopt,
         "datasets/multi_asset/worst_of_call/bs_heston_vasicek_worst_of_calls_01.json",
         "catalog/multi_asset/worst_of_call/bs_heston_vasicek_worst_of_calls_01/generation.yaml",
         "https://datasets.ai-factory.example/v2/multi_asset/worst_of_call/bs_heston_vasicek_worst_of_calls_01.json",
         rate_path},
        bound.model_indices, bound.rows, run.prices, run.standard_errors,
        paths, seed,
        {{"engine", "factorized_cross_asset_mc"},
         {"model_slots", {"black_scholes", "heston"}},
         {"rate_model", "vasicek"},
         {"factor_order", factors},
         {"correlation_factorization", "packed Cholesky FP32"},
         {"random_number_generator", "Philox"}},
        run.wall_seconds, run.kernel_seconds, bound.source_rate_indices);
}
