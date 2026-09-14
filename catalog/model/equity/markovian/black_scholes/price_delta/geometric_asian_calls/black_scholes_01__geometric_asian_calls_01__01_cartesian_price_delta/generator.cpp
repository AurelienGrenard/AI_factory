// Generated Cartesian-product black_scholes geometric_asian_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/black_scholes/product/geometric_asian_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/markovian/black_scholes/price_delta/geometric_asian_calls/black_scholes_01__geometric_asian_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/black_scholes/price_delta/geometric_asian_calls/black_scholes_01__geometric_asian_calls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_delta/geometric_asian_calls/black_scholes_01__geometric_asian_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/black_scholes/prices/geometric_asian_calls/black_scholes_01__geometric_asian_calls_01__01_cartesian/generator.cpp", "centered_closed_form", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_equity_price_delta_dataset<false, false>(
        recipe, {offline::cuda_tuning::PricingFamily::closed_form, "black_scholes", "geometric_asian_option", ""},
        0ULL, model::equity::black_scholes::load_models, product::load_geometric_asian_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::black_scholes::launch_black_scholes_geometric_asian_option_price_delta_cuda<OptionSide::call>(
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
                    1.0f / 504.0f, 2U,
                    context.threads,
                    context.blocks,
                    context.bump,
                    prices,
                    deltas);
        });
}
