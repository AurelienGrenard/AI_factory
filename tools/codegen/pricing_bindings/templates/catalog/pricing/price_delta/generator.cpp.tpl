// Generated $construction_label $model $product price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/$model/product/${product}_price_delta.cuh"
#include "model/equity/markovian/$model/dataset.hpp"
#include "product/$product/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "$model_input", "$product_input", "$dataset", "$catalog",
        "$url", "$source_recipe", "$method", .01, $steps_per_day$construction_argument};
    return pricing::generate_equity_price_delta_dataset<$stochastic, $lsm>(
        recipe, {offline::cuda_tuning::PricingFamily::$family, "$model", "$product", ""},
        ${seed}ULL, model::equity::$model::load_models, $product_loader,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::$model::launch_${model}_${product}_price_delta_cuda$side(
                    $arguments);
        });
}
