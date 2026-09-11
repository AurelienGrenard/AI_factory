// Generated aligned variance_gamma asset_or_nothing_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/variance_gamma/product/asset_or_nothing_option_price_delta.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/markovian/variance_gamma/price_delta/asset_or_nothing_puts/variance_gamma_01__asset_or_nothing_puts_01__01_price_delta.json", "catalog/model/equity/markovian/variance_gamma/price_delta/asset_or_nothing_puts/variance_gamma_01__asset_or_nothing_puts_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/variance_gamma/price_delta/asset_or_nothing_puts/variance_gamma_01__asset_or_nothing_puts_01__01_price_delta.json", "catalog/model/equity/markovian/variance_gamma/prices/asset_or_nothing_puts/variance_gamma_01__asset_or_nothing_puts_01__01/generator.cpp", "centered_crn", .01, 0U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_exact_mc, "variance_gamma", "asset_or_nothing_option", ""},
        11668828120575639552ULL, model::equity::variance_gamma::load_models, product::load_asset_or_nothing_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::variance_gamma::launch_variance_gamma_asset_or_nothing_option_price_delta_cuda<OptionSide::put>(
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
