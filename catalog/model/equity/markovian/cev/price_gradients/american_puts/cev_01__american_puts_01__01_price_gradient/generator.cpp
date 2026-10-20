// Generated cev American selected-gradient recipe with explicit exercise replay.
#include "model/equity/markovian/cev/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/cev/price_gradients/american_puts/cev_01__american_puts_01__01_price_gradient.json", "catalog/model/equity/markovian/cev/price_gradients/american_puts/cev_01__american_puts_01__01_price_gradient/generation.yaml", "catalog/model/equity/markovian/cev/prices/american_puts/cev_01__american_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::frozen_exercise_time;
        recipe.exercise_replay = exercise_replay;
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "cev", "american_option", ""}, 11668826853560287232ULL,
            models, products, model::equity::cev::prepare_cev_american_option_price_gradients,
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::cev::launch_cev_american_option_price_gradients_with_replay_cuda<
                    OptionSide::put>,
                exercise_replay
            ),
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_american_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
