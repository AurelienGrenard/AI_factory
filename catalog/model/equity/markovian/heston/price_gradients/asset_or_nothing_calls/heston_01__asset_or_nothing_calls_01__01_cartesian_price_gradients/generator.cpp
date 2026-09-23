// Generated heston asset_or_nothing_option selected-gradient recipe.
#include "model/equity/markovian/heston/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston/parameters/heston_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/markovian/heston/price_gradients/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/heston/price_gradients/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01_cartesian_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/heston/price_gradients/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/heston/prices/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::heston::load_models(recipe.model_input);
        const auto products = product::load_asset_or_nothing_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "heston", "asset_or_nothing_option", ""}, 11668827008179109888ULL,
            models, products, model::equity::heston::prepare_heston_asset_or_nothing_option_price_gradients,
            model::equity::heston::launch_heston_asset_or_nothing_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::heston::prepare_asset_or_nothing_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
