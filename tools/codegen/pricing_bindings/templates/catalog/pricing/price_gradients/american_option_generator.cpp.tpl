// Generated ${model} American selected-gradient recipe with frozen exercise.
#include "model/equity/markovian/${model}/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/${model}/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog", "$url", "$source_recipe",
            PriceConstruction::$construction, {{
        $selections
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::$model::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "$model", "american_option", ""}, ${seed}ULL,
            models, products, model::equity::$model::prepare_${model}_american_option_price_gradients,
            model::equity::$model::launch_${model}_american_option_price_gradients_cuda<OptionSide::$side>);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
