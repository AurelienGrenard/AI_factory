// Generated ${model} Bermudan selected full Hessian with explicit exercise replay.
#include "model/fixed_income/${model}/product/${curve_prefix}bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
${curve_dataset_include}#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::${binding_namespace};
    try {
        datasets::price_gradients::Recipe recipe{
            "${model_input}", "${product_input}", "${dataset}", "${catalog}",
            "${url}", "${source_recipe}", PriceConstruction::${construction},
            {{
        ${selections}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::$exercise_replay;
        recipe.exercise_replay = exercise_replay;
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
${curve_recipe_assignment}        const auto models =
            model::fixed_income::${model}::load_models(recipe.model_input);
${curve_load}        const auto products =
            product::load_bermudan_swaptions(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            ${curve_lambda_argument}const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_${function_prefix}_bermudan_swaption_sensitivities(
                    model_rows,
                    ${curve_prepare_argument}product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        return offline::pricing::price_gradients::
            ${execute_curve_prefix}mixed_node_graph_dataset<true>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::${family},
                    "${model}",
                    "bermudan_swaption",
                    "${curve}",
                },
                ${seed}ULL,
                models,
                ${execute_curve_argument}products,
                prepare,
                offline::pricing::price_gradients::with_exercise_replay(
                    model_namespace::
                        ${function_prefix}_bermudan_swaption_mixed_node_graph_workspace_bytes_with_replay<
                            SwaptionSide::${side}>,
                    exercise_replay
                ),
                offline::pricing::price_gradients::with_exercise_replay(
                    model_namespace::
                        launch_${function_prefix}_bermudan_swaption_mixed_node_graph_sensitivities_with_replay_cuda<
                            SwaptionSide::${side}>,
                    exercise_replay
                ),
                offline::cuda_tuning::kProductionPathsPerPrice,
                model_namespace::
                    prepare_bermudan_swaption_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
