// Generated rough_heston down_and_in_option paired recipe using the existing host N-factor preparation.
#include "model/equity/rough/rough_heston/product/down_and_in_option_price_delta.cuh"
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"
#include "product/down_and_in_option/dataset.hpp"
#include "tools/pricing/prepared_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json", "datasets/product/down_and_in_option/down_and_in_options_01.json", "datasets/model/equity/rough/rough_heston/price_delta/down_and_in_puts/rough_heston_01__down_and_in_puts_01__01_price_delta.json", "catalog/model/equity/rough/rough_heston/price_delta/down_and_in_puts/rough_heston_01__down_and_in_puts_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_heston/price_delta/down_and_in_puts/rough_heston_01__down_and_in_puts_01__01_price_delta.json", "catalog/model/equity/rough/rough_heston/prices/down_and_in_puts/rough_heston_01__down_and_in_puts_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_prepared_price_delta_dataset(
        recipe, {offline::cuda_tuning::PricingFamily::rough_n_factor, "rough_heston", "down_and_in_option", ""},
        11668828683216355328ULL, 7U, "7-factor Markovian lift", model::equity::rough_heston::load_models, product::load_down_and_in_options,
        [](const auto& models, float horizon) {
            return model::equity::rough_heston::prepare_dynamics<7U>(models, horizon, 1.0f / 504.0f);
        },
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* device_prepared, std::size_t prepared_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            model::equity::rough_heston::launch_rough_heston_down_and_in_option_price_delta_cuda<OptionSide::put, 7U>(
                host_models, device_models, model_count, device_prepared, prepared_count,
                host_products, device_products, product_count, PriceConstruction::Aligned,
                context.results, context.offset, context.count, context.paths,
                1.0f / 504.0f, 2U, context.threads, context.blocks, context.seed,
                context.bump, prices, price_errors, deltas, delta_errors);
        });
}
