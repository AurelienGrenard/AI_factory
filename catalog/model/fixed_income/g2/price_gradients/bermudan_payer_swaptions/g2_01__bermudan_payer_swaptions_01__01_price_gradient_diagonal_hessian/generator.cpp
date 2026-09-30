// Generated g2 Bermudan-swaption gradient and diagonal-Hessian recipe.
#include "model/fixed_income/g2/product/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/g2/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/g2/parameters/g2_01.json", "datasets/product/bermudan_swaption/bermudan_swaptions_01.json", "datasets/model/fixed_income/g2/price_gradients/bermudan_payer_swaptions/g2_01__bermudan_payer_swaptions_01__01_price_gradient_diagonal_hessian.json", "catalog/model/fixed_income/g2/price_gradients/bermudan_payer_swaptions/g2_01__bermudan_payer_swaptions_01__01_price_gradient_diagonal_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/fixed_income/g2/price_gradients/bermudan_payer_swaptions/g2_01__bermudan_payer_swaptions_01__01_price_gradient_diagonal_hessian.json", "catalog/model/fixed_income/g2/prices/bermudan_payer_swaptions/g2_01__bermudan_payer_swaptions_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.mean_reversion_x", {0.005, pg::BumpScale::relative}},
        {"model.volatility_x", {0.005, pg::BumpScale::relative}},
        {"model.mean_reversion_y", {0.005, pg::BumpScale::relative}},
        {"model.volatility_y", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"model.initial_state_x", {0.0005, pg::BumpScale::absolute}},
        {"model.initial_state_y", {0.0005, pg::BumpScale::absolute}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::frozen_exercise_time;
        recipe.exercise_replay = exercise_replay;
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models =
            model::fixed_income::g2::load_models(recipe.model_input);
        const auto products =
            product::load_bermudan_swaptions(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_g2_bermudan_swaption_sensitivities(
                    model_rows, product_rows,
                    construction, time, configuration,
                    {pg::SensitivityOrders::first_and_second}
                );
        };
        return offline::pricing::price_gradients::
            execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::gaussian_rate_lsm,
                "g2",
                "bermudan_swaption",
                "",
            },
            11668829056878510080ULL,
            models,
            products,
            prepare,
            offline::pricing::price_gradients::with_exercise_replay(
                model_namespace::
                    launch_g2_bermudan_swaption_diagonal_sensitivities_with_replay_cuda<
                        SwaptionSide::payer,
                        pg::SensitivityOrders::first_and_second
                    >,
                exercise_replay
            ),
            offline::pricing::price_gradients::with_exercise_replay(
                model_namespace::
                    g2_bermudan_swaption_node_graph_workspace_bytes_with_replay<
                        SwaptionSide::payer,
                        pg::SensitivityOrders::first_and_second
                    >,
                exercise_replay
            ),
            offline::pricing::price_gradients::with_exercise_replay(
                model_namespace::
                    launch_g2_bermudan_swaption_node_graph_sensitivities_with_replay_cuda<
                        SwaptionSide::payer,
                        pg::SensitivityOrders::first_and_second
                    >,
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
