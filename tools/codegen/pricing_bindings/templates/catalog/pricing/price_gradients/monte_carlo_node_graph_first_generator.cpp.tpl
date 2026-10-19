// Generated ${model} ${product} gradient-only recipe.
#include "model/equity/markovian/${model}/product/${product}_price_gradients.cuh"
#include "model/equity/markovian/${model}/dataset.hpp"
#include "product/${product}/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog", "$source_recipe",
            PriceConstruction::$construction, {{
        $selections
            }}, {1.0f/504.0f, 2U}, $exact_transition,
            pg::SensitivityOrders::first};$exact_schedule_assignment$option_side_assignment
        recipe.sensitivity_strategy = pg::SensitivityStrategy::node_graph;
        const auto models = model::equity::$model::load_models(recipe.model_input);
        const auto products = $product_loader_expression;
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::$model::prepare_${model}_${product}_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_first_node_graph_dataset(
            recipe,
            {offline::cuda_tuning::PricingFamily::$family, "$model", "$product", ""},
            ${seed}ULL, models, products, prepare,
            model::equity::$model::${model}_${product}_node_graph_workspace_bytes${sensitivity_template_arguments},
            model::equity::$model::launch_${model}_${product}_node_graph_sensitivities_cuda${sensitivity_template_arguments},
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
