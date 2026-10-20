#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "model/equity/rough/rough_heston/causal_fft_dynamics.cuh"
#include "product/worst_of_call/dataset.hpp"
#include "product/worst_of_call/pricing_policy.cuh"
#include "correlation/dataset.hpp"
#include "tools/datasets/multi_asset_price_dataset.hpp"
#include "tools/pricing/mixed_rough_price_generation.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    using Pack = simulation::MultiAssetPack<simulation::ModelGroup<
        model::equity::black_scholes::DynamicsPolicy, 1U>>;
    using Rough = model::equity::rough_heston::CausalFftPathPolicy;
    using Product = product::WorstOfCallPathPolicy;
    constexpr std::size_t scenarios = 8U;
    const std::filesystem::path bs_path =
        "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json";
    const std::filesystem::path rough_path =
        "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json";
    const std::filesystem::path product_path =
        "datasets/product/worst_of_call/worst_of_calls_01.json";
    const std::filesystem::path correlation_path =
        "datasets/correlation/bs_rough_heston_matrices_01.json";
    const auto bs = model::equity::black_scholes::load_models(bs_path);
    const auto rough = model::equity::rough_heston::load_models(rough_path);
    const auto products = product::load_worst_of_calls(product_path);
    const auto correlations = correlation::load_matrices(correlation_path);
    if (bs.size() < scenarios || rough.size() < scenarios
        || products.size() != 1U || correlations.size() != scenarios) {
        throw std::invalid_argument("Mixed rough reference inputs are incomplete.");
    }
    offline::pricing::MultiAssetHostModels<Pack> host;
    host.models = bs;
    std::vector<std::uint32_t> rough_indices;
    std::vector<std::uint32_t> scenario_model_indices;
    std::vector<monte_carlo::MultiAssetPriceRow> rows;
    for (std::uint32_t scenario = 0U; scenario < scenarios; ++scenario) {
        host.scenario_indices.push_back(scenario);
        rough_indices.push_back(scenario);
        scenario_model_indices.push_back(scenario);
        scenario_model_indices.push_back(scenario);
        rows.push_back({scenario, 0U, scenario, 0U,
            std::exp(-bs[scenario].risk_free_rate
                * static_cast<float>(products[0].maturity_days) / 252.0f)});
    }
    constexpr std::size_t paths = 4096U;
    constexpr std::uint32_t steps_per_day = 2U;
    constexpr float dt = 1.0f / (252.0f * steps_per_day);
    const std::size_t chunk = argc >= 2 ? std::stoul(argv[1]) : paths;
    constexpr std::uint64_t seed = 20261019ULL;
    const auto run = offline::pricing::execute_mixed_rough_prices<
        Pack, Rough, Product
    >(
        host, rough, rough_indices, scenarios, products, correlations,
        {"asset_0.spot", "asset_1.spot", "asset_1.vol"},
        {"asset_1.vol", "asset_0.spot", "asset_1.spot"},
        rows, paths, chunk, dt, steps_per_day, 32U, seed
    );
    datasets::write_multi_asset_price_dataset(
        {{bs_path, rough_path}, product_path, correlation_path,
         std::nullopt,
         "datasets/multi_asset/worst_of_call/bs_rough_heston_worst_of_calls_01.json",
         "catalog/multi_asset/worst_of_call/bs_rough_heston_worst_of_calls_01/generation.yaml",
         "https://datasets.ai-factory.example/v2/multi_asset/worst_of_call/bs_rough_heston_worst_of_calls_01.json"},
        scenario_model_indices, rows, run.prices, run.standard_errors,
        paths, seed,
        {{"engine", "multi_asset_causal_fft"},
         {"strategy", "volatility_first"},
         {"model_slots", {"black_scholes", "rough_heston"}},
         {"factor_order", {"asset_1.vol", "asset_0.spot", "asset_1.spot"}},
         {"direct_tile_size", 32U},
         {"path_chunk_size", chunk},
         {"correlation_factorization", "packed Cholesky FP32"},
         {"random_number_generator", "Philox"}},
        run.wall_seconds, run.kernel_seconds
    );
}
