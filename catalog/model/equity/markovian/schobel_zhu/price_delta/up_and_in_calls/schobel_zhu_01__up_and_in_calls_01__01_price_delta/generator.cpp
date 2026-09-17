// Generated aligned schobel_zhu up_and_in_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/schobel_zhu/product/up_and_in_option_price_delta.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/up_and_in_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/up_and_in_option/up_and_in_options_01.json", "datasets/model/equity/markovian/schobel_zhu/price_delta/up_and_in_calls/schobel_zhu_01__up_and_in_calls_01__01_price_delta.json", "catalog/model/equity/markovian/schobel_zhu/price_delta/up_and_in_calls/schobel_zhu_01__up_and_in_calls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/schobel_zhu/price_delta/up_and_in_calls/schobel_zhu_01__up_and_in_calls_01__01_price_delta.json", "catalog/model/equity/markovian/schobel_zhu/prices/up_and_in_calls/schobel_zhu_01__up_and_in_calls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "schobel_zhu", "up_and_in_option", ""},
        11668827940187013120ULL, model::equity::schobel_zhu::load_models, product::load_up_and_in_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::schobel_zhu::launch_schobel_zhu_up_and_in_option_price_delta_cuda<OptionSide::call>(
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
