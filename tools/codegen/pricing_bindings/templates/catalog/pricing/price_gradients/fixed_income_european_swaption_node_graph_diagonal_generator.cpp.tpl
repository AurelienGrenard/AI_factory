// Generated ${model} European-swaption MC gradient and diagonal-Hessian recipe.
#include "model/fixed_income/${model}/product/${curve_prefix}european_swaption_price_gradients.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
${curve_dataset_include}#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::${binding_namespace};
    try {
        datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog",
            "$url", "$source_recipe", PriceConstruction::$construction,
            {{
        $selections
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
${curve_recipe_assignment}        const auto models =
            model::fixed_income::${model}::load_models(recipe.model_input);
${curve_load}        const auto product_dataset =
            product::load_european_swaptions(recipe.product_input);
        const auto& products = product_dataset.products;
        const auto prepare = [](
            const auto& model_rows,
            ${curve_lambda_argument}const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_${function_prefix}_european_swaption_sensitivities(
                    model_rows, ${curve_prepare_argument}product_rows,
                    construction, time, configuration,
                    {pg::SensitivityOrders::first_and_second}
                );
        };
        return offline::pricing::price_gradients::
            ${execute_curve_prefix}node_graph_dataset<
                pg::SensitivityOrders::first_and_second
            >(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::$family,
                    "$model",
                    "european_swaption",
                    "$curve",
                },
                ${seed}ULL,
                models,
                ${execute_curve_argument}products,
                prepare,
                model_namespace::
                    launch_${function_prefix}_european_swaption_diagonal_sensitivities_cuda<
                        SwaptionSide::$side,
                        pg::SensitivityOrders::first_and_second
                    >,
                model_namespace::
                    ${function_prefix}_european_swaption_node_graph_workspace_bytes<
                        SwaptionSide::$side,
                        pg::SensitivityOrders::first_and_second
                    >,
                model_namespace::
                    launch_${function_prefix}_european_swaption_node_graph_sensitivities_cuda<
                        SwaptionSide::$side,
                        pg::SensitivityOrders::first_and_second
                    >,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model_namespace::
                    prepare_european_swaption_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
