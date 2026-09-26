// Generated ${model} European-swaption selected full-Hessian recipe.
#include "model/fixed_income/${model}/product/${curve_prefix}european_swaption_price_gradients.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
${curve_dataset_include}#include "product/european_swaption/dataset.hpp"
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
${curve_load}        const auto product_dataset =
            product::load_european_swaptions(recipe.product_input);
        const auto& products = product_dataset.products;
        const auto prepare = [](
            const auto& model_rows,
            ${curve_lambda_argument}const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_${function_prefix}_european_swaption_sensitivities(
                    model_rows,
                    ${curve_prepare_argument}product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        const auto launch = [](
            const auto& plan,
            auto inputs,
            auto stencils,
            auto mixed_stencils,
            const auto& configuration,
            auto outputs,
            auto mixed_outputs,
            void* workspace,
            std::size_t workspace_bytes
        ) {
            return model_namespace::
                launch_${function_prefix}_european_swaption_mixed_node_graph_sensitivities_cuda<
                    SwaptionSide::${side}>(
                        plan,
                        inputs,
                        stencils,
                        mixed_stencils,
                        configuration,
                        outputs,
                        mixed_outputs,
                        workspace,
                        workspace_bytes${mixed_distribution_argument}
                    );
        };
        return offline::pricing::price_gradients::
            ${execute_curve_prefix}mixed_node_graph_dataset<${stochastic}>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::${family},
                    "${model}",
                    "european_swaption",
                    "${curve}",
                },
                ${seed}ULL,
                models,
                ${execute_curve_argument}products,
                prepare,
                model_namespace::
                    ${function_prefix}_european_swaption_mixed_node_graph_workspace_bytes<
                        SwaptionSide::${side}>,
                launch,
                ${paths_per_price},
                model_namespace::
                    prepare_european_swaption_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
