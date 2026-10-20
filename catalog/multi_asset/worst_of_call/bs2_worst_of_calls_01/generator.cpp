// Two Black-Scholes slots with a single correlated Brownian price engine.
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "product/worst_of_call/dataset.hpp"
#include "product/worst_of_call/pricing_policy.cuh"
#include "correlation/dataset.hpp"
#include "tools/datasets/multi_asset_price_dataset.hpp"
#include "tools/pricing/multi_asset_price_generation.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    using Pack = simulation::MultiAssetPack<
        simulation::ModelGroup<
            model::equity::black_scholes::DynamicsPolicy, 2U>
    >;
    using Product = product::WorstOfCallPathPolicy;
    constexpr std::size_t scenario_count = 32U;
    constexpr std::uint32_t steps_per_day = 2U;
    constexpr float dt = 1.0f / (252.0f * steps_per_day);
    const unsigned int team_size = argc == 2
        ? static_cast<unsigned int>(std::stoul(argv[1]))
        : 1U;

    const std::filesystem::path model_path =
        "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json";
    const std::filesystem::path product_path =
        "datasets/product/worst_of_call/worst_of_calls_01.json";
    const std::filesystem::path correlation_path =
        "datasets/correlation/correlation_matrices_01.json";
    const auto models = model::equity::black_scholes::load_models(model_path);
    const auto products = product::load_worst_of_calls(product_path);
    const auto correlations = correlation::load_matrices(correlation_path);
    if (models.size() < scenario_count || products.size() != 1U
        || correlations.size() != 1U) {
        throw std::invalid_argument("Reference multi-asset inputs are incomplete.");
    }

    offline::pricing::MultiAssetHostModels<Pack> host;
    host.models = models;
    std::vector<std::uint32_t> scenario_model_indices;
    std::vector<monte_carlo::MultiAssetPriceRow> rows;
    for (std::uint32_t scenario = 0U; scenario < scenario_count;
         ++scenario) {
        host.scenario_indices.push_back(scenario);
        host.scenario_indices.push_back(scenario);
        scenario_model_indices.push_back(scenario);
        scenario_model_indices.push_back(scenario);
        rows.push_back({
            scenario, 0U, 0U, 0U,
            std::exp(-models[scenario].risk_free_rate
                * static_cast<float>(products[0].maturity_days)
                / 252.0f)
        });
    }

    offline::pricing::BatchedMonteCarloProfile profile{};
    profile.paths_per_price = 16384U;
    profile.results_per_launch = 32U;
    profile.block_count_limit = 32U;
    profile.threads_per_block = 128U;
    profile.seed = 20261019ULL;
    profile.delta_t_description = "1 / 504";

    const auto run = [&]() -> offline::cuda::MonteCarloRun {
        switch (team_size) {
        case 1U:
            return offline::pricing::execute_multi_asset_prices<
                Pack, Product, 1U
            >(host, scenario_count, products, correlations,
              {"asset_0.spot", "asset_1.spot"}, rows,
              profile, dt, steps_per_day);
        case 2U:
            return offline::pricing::execute_multi_asset_prices<
                Pack, Product, 2U
            >(host, scenario_count, products, correlations,
              {"asset_0.spot", "asset_1.spot"}, rows,
              profile, dt, steps_per_day);
        case 4U:
            return offline::pricing::execute_multi_asset_prices<
                Pack, Product, 4U
            >(host, scenario_count, products, correlations,
              {"asset_0.spot", "asset_1.spot"}, rows,
              profile, dt, steps_per_day);
        case 8U:
            return offline::pricing::execute_multi_asset_prices<
                Pack, Product, 8U
            >(host, scenario_count, products, correlations,
              {"asset_0.spot", "asset_1.spot"}, rows,
              profile, dt, steps_per_day);
        case 16U:
            return offline::pricing::execute_multi_asset_prices<
                Pack, Product, 16U
            >(host, scenario_count, products, correlations,
              {"asset_0.spot", "asset_1.spot"}, rows,
              profile, dt, steps_per_day);
        case 32U:
            return offline::pricing::execute_multi_asset_prices<
                Pack, Product, 32U
            >(host, scenario_count, products, correlations,
              {"asset_0.spot", "asset_1.spot"}, rows,
              profile, dt, steps_per_day);
        default:
            throw std::invalid_argument("Team size must be 1, 2, 4, 8, 16 or 32.");
        }
    }();

    datasets::write_multi_asset_price_dataset(
        {
            {model_path, model_path}, product_path, correlation_path,
            std::nullopt,
            "datasets/multi_asset/worst_of_call/bs2_worst_of_calls_01.json",
            "catalog/multi_asset/worst_of_call/bs2_worst_of_calls_01/generation.yaml",
            "https://datasets.ai-factory.example/v2/multi_asset/worst_of_call/bs2_worst_of_calls_01.json"
        },
        scenario_model_indices, rows,
        run.prices, run.standard_errors, profile.paths_per_price,
        profile.seed,
        {
            {"engine", "multi_asset_mc"},
            {"model_slots", {"black_scholes", "black_scholes"}},
            {"team_size", team_size},
            {"threads_per_block", profile.threads_per_block},
            {"correlation_factorization", "packed Cholesky FP32"},
            {"random_number_generator", "Philox"}
        },
        run.wall_seconds, run.kernel_seconds
    );
    return 0;
}
