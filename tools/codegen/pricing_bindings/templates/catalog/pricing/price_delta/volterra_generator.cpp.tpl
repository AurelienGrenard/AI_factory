// Generated $model $product paired recipe with one shared FFT workspace.
#include "model/equity/rough/$model/product/${product}_price_delta.cuh"
#include "model/equity/rough/$model/dataset.hpp"
#include "product/$product/dataset.hpp"
#include "product/$product/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "$model_input", "$product_input", "$dataset", "$catalog",
        "$url", "$source_recipe", "$method", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<$schedule, $product_policy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "$model", "$product", ""},
        ${seed}ULL, "$numerical_method", model::equity::$model::load_models, $product_loader,
        [](auto... arguments) {
            model::equity::$model::launch_${model}_${product}_price_delta_cuda$side(arguments...);
        });
}
