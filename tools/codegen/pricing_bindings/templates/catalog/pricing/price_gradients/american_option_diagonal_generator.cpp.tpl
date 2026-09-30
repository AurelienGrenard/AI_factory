// Generated ${model} American gradient and diagonal Hessian with explicit exercise replay.
#include "model/equity/markovian/${model}/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/${model}/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog", "$url", "$source_recipe",
            PriceConstruction::$construction, {{
        $selections
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::$exercise_replay;
        recipe.exercise_replay = exercise_replay;
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::$model::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::$model::prepare_${model}_american_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "$model", "american_option", ""},
            ${seed}ULL, models, products, prepare,
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::$model::launch_${model}_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::$side, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::$model::${model}_american_option_node_graph_workspace_bytes_with_replay<
                    OptionSide::$side, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::$model::launch_${model}_american_option_node_graph_sensitivities_with_replay_cuda<
                    OptionSide::$side, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::$model::prepare_american_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
