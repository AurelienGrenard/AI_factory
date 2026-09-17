// Generated aligned heston_3_2 european_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/heston_3_2/product/european_option_price_delta.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/markovian/heston_3_2/price_delta/european_puts/heston_3_2_01__european_puts_01__01_price_delta.json", "catalog/model/equity/markovian/heston_3_2/price_delta/european_puts/heston_3_2_01__european_puts_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/heston_3_2/price_delta/european_puts/heston_3_2_01__european_puts_01__01_price_delta.json", "catalog/model/equity/markovian/heston_3_2/prices/european_puts/heston_3_2_01__european_puts_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "heston_3_2", "european_option", ""},
        11668827188567736320ULL, model::equity::heston_3_2::load_models, product::load_european_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::heston_3_2::launch_heston_3_2_european_option_price_delta_cuda<OptionSide::put>(
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
