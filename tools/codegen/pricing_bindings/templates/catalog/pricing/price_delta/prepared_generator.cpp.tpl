// Generated $construction_prefix$model $product paired recipe using the existing host N-factor preparation.
#include "model/equity/rough/$model/product/${product}_price_delta.cuh"
#include "model/equity/rough/$model/dataset.hpp"
#include "model/equity/rough/$model/markovian_n_factor_preparation.hpp"
#include "product/$product/dataset.hpp"
#include "tools/pricing/prepared_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "$model_input", "$product_input", "$dataset", "$catalog",
        "$url", "$source_recipe", "$method", .01, 2U$construction_argument};
    return pricing::generate_prepared_price_delta_dataset(
        recipe, {offline::cuda_tuning::PricingFamily::rough_n_factor, "$model", "$product", ""},
        ${seed}ULL, 7U, "$numerical_method", model::equity::$model::load_models, $product_loader,
        [](const auto& models, float horizon) {
            return model::equity::$model::prepare_dynamics<7U>(models, horizon, 1.0f / 504.0f);
        },
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* device_prepared, std::size_t prepared_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            model::equity::$model::launch_${model}_${product}_price_delta_cuda<$template_arguments>(
                host_models, device_models, model_count, device_prepared, prepared_count,
                host_products, device_products, product_count, PriceConstruction::$construction,
                context.results, context.offset, context.count, context.paths,
                1.0f / 504.0f, 2U, context.threads, context.blocks, context.seed,
                context.bump, prices, price_errors, deltas, delta_errors);
        });
}
