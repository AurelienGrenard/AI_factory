// Generated ${model} ${product} selected-gradient recipe.
#include "model/equity/markovian/${model}/product/${product}_price_gradients.cuh"
#include "model/equity/markovian/${model}/dataset.hpp"
#include "product/${product}/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog", "$url", "$source_recipe",
            PriceConstruction::$construction, {{
        $selections
            }}, {1.0f/504.0f, 2U}, $exact_transition};
        const auto models = model::equity::$model::load_models(recipe.model_input);
        const auto products = $product_loader_expression;
        return offline::pricing::price_gradients::execute_dataset<$stochastic>(recipe,
            {offline::cuda_tuning::PricingFamily::$family, "$model", "$product", ""}, ${seed}ULL,
            models, products, model::equity::$model::prepare_${model}_${product}_price_gradients,
            model::equity::$model::launch_${model}_${product}_price_gradients_cuda${price_template_arguments},
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::$model::prepare_${product}_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
