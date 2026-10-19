// Generated schobel_zhu forward_start_option selected-gradient recipe.
#include "model/equity/markovian/schobel_zhu/product/forward_start_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/markovian/schobel_zhu/price_gradients/forward_start_calls/schobel_zhu_01__forward_start_calls_01__01_price_gradient.json", "catalog/model/equity/markovian/schobel_zhu/price_gradients/forward_start_calls/schobel_zhu_01__forward_start_calls_01__01_price_gradient/generation.yaml", "catalog/model/equity/markovian/schobel_zhu/prices/forward_start_calls/schobel_zhu_01__forward_start_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_run_volatility", {0.005, pg::BumpScale::relative}},
        {"model.volatility_of_volatility", {0.005, pg::BumpScale::relative}},
        {"product.moneyness", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false};
        recipe.option_side = "call";
        const auto models = model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_forward_start_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "schobel_zhu", "forward_start_option", ""}, 11668827892942372864ULL,
            models, products, model::equity::schobel_zhu::prepare_schobel_zhu_forward_start_option_price_gradients,
            model::equity::schobel_zhu::launch_schobel_zhu_forward_start_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::schobel_zhu::prepare_forward_start_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
