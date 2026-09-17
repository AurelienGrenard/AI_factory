// Generated Cartesian-product stein_stein asset_or_nothing_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/stein_stein/product/asset_or_nothing_option_price_delta.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/markovian/stein_stein/price_delta/asset_or_nothing_calls/stein_stein_01__asset_or_nothing_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/stein_stein/price_delta/asset_or_nothing_calls/stein_stein_01__asset_or_nothing_calls_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/stein_stein/price_delta/asset_or_nothing_calls/stein_stein_01__asset_or_nothing_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/stein_stein/prices/asset_or_nothing_calls/stein_stein_01__asset_or_nothing_calls_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "stein_stein", "asset_or_nothing_option", ""},
        11668827974546751488ULL, model::equity::stein_stein::load_models, product::load_asset_or_nothing_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::stein_stein::launch_stein_stein_asset_or_nothing_option_price_delta_cuda<OptionSide::call>(
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
                    1.0f / 504.0f, 2U,
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
