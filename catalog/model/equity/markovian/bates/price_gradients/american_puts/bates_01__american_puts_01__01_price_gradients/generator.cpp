// Generated bates American selected-gradient recipe with frozen exercise.
#include "model/equity/markovian/bates/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/bates/price_gradients/american_puts/bates_01__american_puts_01__01_price_gradients.json", "catalog/model/equity/markovian/bates/price_gradients/american_puts/bates_01__american_puts_01__01_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_gradients/american_puts/bates_01__american_puts_01__01_price_gradients.json", "catalog/model/equity/markovian/bates/prices/american_puts/bates_01__american_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "bates", "american_option", ""}, 11668826638811922432ULL,
            models, products, model::equity::bates::prepare_bates_american_option_price_gradients,
            model::equity::bates::launch_bates_american_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_american_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
