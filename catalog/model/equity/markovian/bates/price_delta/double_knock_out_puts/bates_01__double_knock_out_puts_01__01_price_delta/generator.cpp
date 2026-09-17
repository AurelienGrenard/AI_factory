// Generated aligned bates double_knock_out_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/bates/product/double_knock_out_option_price_delta.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/double_knock_out_option/double_knock_out_options_01.json", "datasets/model/equity/markovian/bates/price_delta/double_knock_out_puts/bates_01__double_knock_out_puts_01__01_price_delta.json", "catalog/model/equity/markovian/bates/price_delta/double_knock_out_puts/bates_01__double_knock_out_puts_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/bates/price_delta/double_knock_out_puts/bates_01__double_knock_out_puts_01__01_price_delta.json", "catalog/model/equity/markovian/bates/prices/double_knock_out_puts/bates_01__double_knock_out_puts_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "double_knock_out_option", ""},
        11668826681761595392ULL, model::equity::bates::load_models, product::load_double_knock_out_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::bates::launch_bates_double_knock_out_option_price_delta_cuda<OptionSide::put>(
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
