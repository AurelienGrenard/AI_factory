// Generated ${model} ${product} selected full-Hessian recipe.
#include "model/fixed_income/${model}/product/${curve_prefix}${product}_price_gradients.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
${curve_dataset_include}#include "product/${product}/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::${binding_namespace};
    try {
        datasets::price_gradients::Recipe recipe{
            "${model_input}", "${product_input}", "${dataset}", "${catalog}",
            "${url}", "${source_recipe}", PriceConstruction::${construction},
            {{
        ${selections}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
${curve_recipe_assignment}        const auto models =
            model::fixed_income::${model}::load_models(recipe.model_input);
${curve_load}        const auto products =
            product::${product_loader}(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            ${curve_lambda_argument}const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_${function_prefix}_${product}_sensitivities(
                    model_rows,
                    ${curve_prepare_argument}product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        return offline::pricing::price_gradients::
            ${execute_curve_prefix}mixed_node_graph_dataset<false>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::closed_form,
                    "${model}",
                    "${product}",
                    "${curve}",
                },
                0ULL,
                models,
                ${execute_curve_argument}products,
                prepare,
                model_namespace::
                    ${function_prefix}_${product}_mixed_node_graph_workspace_bytes<
                        OptionSide::${side}>,
                model_namespace::
                    launch_${function_prefix}_${product}_mixed_node_graph_sensitivities_cuda<
                        OptionSide::${side}>,
                0U,
                model_namespace::
                    prepare_${product}_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
