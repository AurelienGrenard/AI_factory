// Generated schobel_zhu American selected-gradient recipe with explicit exercise replay.
#include "model/equity/markovian/schobel_zhu/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/schobel_zhu/price_gradients/american_calls/schobel_zhu_01__american_calls_01__01_price_gradient.json", "catalog/model/equity/markovian/schobel_zhu/price_gradients/american_calls/schobel_zhu_01__american_calls_01__01_price_gradient/generation.yaml", "catalog/model/equity/markovian/schobel_zhu/prices/american_calls/schobel_zhu_01__american_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_run_volatility", {0.005, pg::BumpScale::relative}},
        {"model.volatility_of_volatility", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::frozen_exercise_time;
        recipe.exercise_replay = exercise_replay;
        const auto models = model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "schobel_zhu", "american_option", ""}, 11668827824222896128ULL,
            models, products, model::equity::schobel_zhu::prepare_schobel_zhu_american_option_price_gradients,
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::schobel_zhu::launch_schobel_zhu_american_option_price_gradients_with_replay_cuda<
                    OptionSide::call>,
                exercise_replay
            ),
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::schobel_zhu::prepare_american_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
