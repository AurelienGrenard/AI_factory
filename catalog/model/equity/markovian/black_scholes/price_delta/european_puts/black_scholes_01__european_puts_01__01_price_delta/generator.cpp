// Generated aligned black_scholes european_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/black_scholes/product/european_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/markovian/black_scholes/price_delta/european_puts/black_scholes_01__european_puts_01__01_price_delta.json", "catalog/model/equity/markovian/black_scholes/price_delta/european_puts/black_scholes_01__european_puts_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_delta/european_puts/black_scholes_01__european_puts_01__01_price_delta.json", "catalog/model/equity/markovian/black_scholes/prices/european_puts/black_scholes_01__european_puts_01__01/recipe.yaml", "centered_closed_form", .01, 0U};
    return pricing::generate_equity_price_delta_dataset<false, false>(
        recipe, {offline::cuda_tuning::PricingFamily::closed_form, "black_scholes", "european_option", ""},
        0ULL, model::equity::black_scholes::load_models, product::load_european_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::black_scholes::launch_black_scholes_european_option_price_delta_cuda<OptionSide::put>(
                    host_models,
                    device_models,
                    model_count,
                    device_products,
                    product_count,
                    PriceConstruction::Aligned,
                    context.results,
                    context.offset,
                    context.count,
                    1.0f / 252.0f,
                    context.threads,
                    context.blocks,
                    context.bump,
                    prices,
                    deltas);
        });
}
