// Generated ${model} ${product} gradient and diagonal-Hessian recipe.
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
            }}, {1.0f/504.0f, 2U}, $exact_transition,
            pg::SensitivityOrders::first_and_second};
        const auto models = model::equity::$model::load_models(recipe.model_input);
        const auto products = $product_loader_expression;
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::$model::prepare_${model}_${product}_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_dataset<
            $stochastic, pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::$family, "$model", "$product", ""},
            ${seed}ULL, models, products, prepare,
            model::equity::$model::launch_${model}_${product}_diagonal_sensitivities_cuda<
                OptionSide::$side, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::$model::prepare_${product}_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
