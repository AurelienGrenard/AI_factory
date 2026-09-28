// Generated merton asset_or_nothing_option selected-gradient recipe.
#include "model/equity/markovian/merton/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/markovian/merton/price_sensitivities/asset_or_nothing_puts/merton_01__asset_or_nothing_puts_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/merton/price_sensitivities/asset_or_nothing_puts/merton_01__asset_or_nothing_puts_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_sensitivities/asset_or_nothing_puts/merton_01__asset_or_nothing_puts_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/merton/prices/asset_or_nothing_puts/merton_01__asset_or_nothing_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_asset_or_nothing_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "merton", "asset_or_nothing_option", ""}, 11668827429085904896ULL,
            models, products, model::equity::merton::prepare_merton_asset_or_nothing_option_price_gradients,
            model::equity::merton::launch_merton_asset_or_nothing_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::merton::prepare_asset_or_nothing_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
