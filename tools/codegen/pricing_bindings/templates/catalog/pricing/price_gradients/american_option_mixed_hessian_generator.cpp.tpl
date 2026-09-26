// Generated ${model} American selected full Hessian with frozen exercise.
#include "model/equity/markovian/${model}/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/${model}/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "${model_input}", "${product_input}", "${dataset}", "${catalog}",
            "${url}", "${source_recipe}", PriceConstruction::${construction},
            {{
        ${selections}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::${model}::load_models(recipe.model_input);
        const auto products =
            product::load_american_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::${model}::
                prepare_${model}_american_option_sensitivities(
                    model_rows,
                    product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        return offline::pricing::price_gradients::
            execute_mixed_node_graph_dataset<true>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::equity_lsm,
                    "${model}",
                    "american_option",
                    "",
                },
                ${seed}ULL,
                models,
                products,
                prepare,
                model::equity::${model}::
                    ${model}_american_option_mixed_node_graph_workspace_bytes<
                        OptionSide::${side}>,
                model::equity::${model}::
                    launch_${model}_american_option_mixed_node_graph_sensitivities_cuda<
                        OptionSide::${side}>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::${model}::
                    prepare_american_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
