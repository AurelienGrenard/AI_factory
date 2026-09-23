// Generated aligned black_scholes cliquet price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/black_scholes/product/cliquet_price_delta.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/markovian/black_scholes/price_delta/cliquets/black_scholes_01__cliquets_01__01_price_delta.json", "catalog/model/equity/markovian/black_scholes/price_delta/cliquets/black_scholes_01__cliquets_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_delta/cliquets/black_scholes_01__cliquets_01__01_price_delta.json", "catalog/model/equity/markovian/black_scholes/prices/cliquets/black_scholes_01__cliquets_01__01/recipe.yaml", "centered_crn", .01, 0U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_exact_mc, "black_scholes", "cliquet", ""},
        11668826789135777792ULL, model::equity::black_scholes::load_models, product::load_cliquets,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::black_scholes::launch_black_scholes_cliquet_price_delta_cuda(
                    host_models,
                    device_models,
                    model_count,
                    host_products,
                    device_products,
                    product_count,
                    PriceConstruction::Aligned,
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
