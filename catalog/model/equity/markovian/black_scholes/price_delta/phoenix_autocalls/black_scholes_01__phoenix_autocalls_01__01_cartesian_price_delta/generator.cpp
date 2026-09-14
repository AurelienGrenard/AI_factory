// Generated Cartesian-product black_scholes phoenix_autocall price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/black_scholes/product/phoenix_autocall_price_delta.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/black_scholes/price_delta/phoenix_autocalls/black_scholes_01__phoenix_autocalls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/black_scholes/price_delta/phoenix_autocalls/black_scholes_01__phoenix_autocalls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_delta/phoenix_autocalls/black_scholes_01__phoenix_autocalls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/black_scholes/prices/phoenix_autocalls/black_scholes_01__phoenix_autocalls_01__01_cartesian/generator.cpp", "centered_crn", .01, 0U, PriceConstruction::CartesianProduct};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_exact_mc, "black_scholes", "phoenix_autocall", ""},
        11668826814905581568ULL, model::equity::black_scholes::load_models, product::load_phoenix_autocalls,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::black_scholes::launch_black_scholes_phoenix_autocall_price_delta_cuda(
                    host_models,
                    device_models,
                    model_count,
                    host_products,
                    device_products,
                    product_count,
                    PriceConstruction::CartesianProduct,
                    context.results,
                    context.offset,
                    context.count,
                    context.paths,
                    1.0f / 252.0f,
                    context.threads,
                    context.blocks,
                    context.seed,
                    context.bump,
                    prices,
                    price_errors,
                    deltas,
                    delta_errors);
        });
}
