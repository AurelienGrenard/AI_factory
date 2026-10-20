// Generated normal_inverse_gaussian American selected-gradient recipe with explicit exercise replay.
#include "model/equity/markovian/normal_inverse_gaussian/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_gradients/american_puts/normal_inverse_gaussian_01__american_puts_01__01_price_gradient.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_gradients/american_puts/normal_inverse_gaussian_01__american_puts_01__01_price_gradient/generation.yaml", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/american_puts/normal_inverse_gaussian_01__american_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.alpha", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.016, pg::BumpScale::absolute}},
        {"model.delta", {0.005, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::frozen_exercise_time;
        recipe.exercise_replay = exercise_replay;
        const auto models = model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "normal_inverse_gaussian", "american_option", ""}, 11668827553639956480ULL,
            models, products, model::equity::normal_inverse_gaussian::prepare_normal_inverse_gaussian_american_option_price_gradients,
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_american_option_price_gradients_with_replay_cuda<
                    OptionSide::put>,
                exercise_replay
            ),
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::normal_inverse_gaussian::prepare_american_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
