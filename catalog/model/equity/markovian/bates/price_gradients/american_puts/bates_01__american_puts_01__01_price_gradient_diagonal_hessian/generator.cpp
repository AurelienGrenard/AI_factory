// Generated bates American gradient and diagonal Hessian with explicit exercise replay.
#include "model/equity/markovian/bates/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/bates/price_gradients/american_puts/bates_01__american_puts_01__01_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/bates/price_gradients/american_puts/bates_01__american_puts_01__01_price_gradient_diagonal_hessian/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_gradients/american_puts/bates_01__american_puts_01__01_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/bates/prices/american_puts/bates_01__american_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::frozen_exercise_time;
        recipe.exercise_replay = exercise_replay;
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::bates::prepare_bates_american_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "bates", "american_option", ""},
            11668826638811922432ULL, models, products, prepare,
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::bates::launch_bates_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::put, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::bates::bates_american_option_node_graph_workspace_bytes_with_replay<
                    OptionSide::put, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::bates::launch_bates_american_option_node_graph_sensitivities_with_replay_cuda<
                    OptionSide::put, pg::SensitivityOrders::first_and_second>,
                exercise_replay
            ),
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_american_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
