// Generated schobel_zhu American gradient and diagonal Hessian with explicit exercise replay.
#include "model/equity/markovian/schobel_zhu/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/schobel_zhu/price_gradients/american_calls/schobel_zhu_01__american_calls_01__01_price_gradient_diagonal_hessian_frozen_policy.json", "catalog/model/equity/markovian/schobel_zhu/price_gradients/american_calls/schobel_zhu_01__american_calls_01__01_price_gradient_diagonal_hessian_frozen_policy/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/price_gradients/american_calls/schobel_zhu_01__american_calls_01__01_price_gradient_diagonal_hessian_frozen_policy.json", "catalog/model/equity/markovian/schobel_zhu/prices/american_calls/schobel_zhu_01__american_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::frozen_regression_policy;
        recipe.exercise_replay = exercise_replay;
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::schobel_zhu::prepare_schobel_zhu_american_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "schobel_zhu", "american_option", ""},
            11668827824222896128ULL, models, products, prepare,
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::schobel_zhu::launch_schobel_zhu_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::call, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::schobel_zhu::schobel_zhu_american_option_node_graph_workspace_bytes_with_replay<
                    OptionSide::call, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::schobel_zhu::launch_schobel_zhu_american_option_node_graph_sensitivities_with_replay_cuda<
                    OptionSide::call, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::schobel_zhu::prepare_american_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
