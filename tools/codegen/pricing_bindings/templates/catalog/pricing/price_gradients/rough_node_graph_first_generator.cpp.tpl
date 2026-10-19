// Generated ${model} ${product} ${rough_construction_label}first-order rough graph recipe.
#include "model/equity/rough/${model}/product/${product}_price_gradients.cuh"
#include "model/equity/rough/${model}/product/${product}.cuh"
#include "model/equity/rough/${model}/dataset.hpp"
#include "product/${product}/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::$model;
    try {
        datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog",
            "$source_recipe", PriceConstruction::$construction, {{
        $selections
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };$option_side_assignment
        const auto models = binding::load_models(recipe.model_input);
        const auto products = $product_loader_expression;
        using Graph = binding::${rough_product_type}NodeGraph<
            $rough_graph_template_arguments>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_${model}_${product}_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::$family,
             "$model", "$product", ""},
            ${seed}ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
