// Generated ${model} European selected-gradient recipe using the shared native runner.
#include "model/equity/markovian/${model}/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/${model}/dataset.hpp"
#include "product/european_option/dataset.hpp"
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
        const auto products = product::load_european_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<$stochastic>(recipe,
            {offline::cuda_tuning::PricingFamily::$family, "$model", "european_option", ""}, ${seed}ULL,
            models, products, model::equity::$model::prepare_${model}_european_option_price_gradients,
            model::equity::$model::launch_${model}_european_option_price_gradients_cuda<OptionSide::$side>);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
