// Generated ${model} ${product} gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/${model}/product/${product}_price_gradients.cuh"
#include "model/equity/markovian/${model}/dataset.hpp"
#include "product/${product}/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog", "$url", "$source_recipe",
            PriceConstruction::$construction, {{
        $selections
            }}, {1.0f/504.0f, 2U}, $exact_transition,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::$model::load_models(recipe.model_input);
        const auto products = $product_loader_expression;
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::$model::prepare_${model}_${product}_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::$family, "$model", "$product", ""},
            ${seed}ULL, models, products, prepare,
            model::equity::$model::launch_${model}_${product}_diagonal_sensitivities_cuda${sensitivity_template_arguments},
            model::equity::$model::${model}_${product}_node_graph_workspace_bytes${sensitivity_template_arguments},
            model::equity::$model::launch_${model}_${product}_node_graph_sensitivities_cuda${sensitivity_template_arguments},
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::$model::prepare_${product}_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
