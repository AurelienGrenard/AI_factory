// Generated schobel_zhu cliquet selected-gradient recipe.
#include "model/equity/markovian/schobel_zhu/product/cliquet_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/markovian/schobel_zhu/price_gradients/cliquets/schobel_zhu_01__cliquets_01__01_price_gradient.json", "catalog/model/equity/markovian/schobel_zhu/price_gradients/cliquets/schobel_zhu_01__cliquets_01__01_price_gradient/generation.yaml", "catalog/model/equity/markovian/schobel_zhu/prices/cliquets/schobel_zhu_01__cliquets_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_run_volatility", {0.005, pg::BumpScale::relative}},
        {"model.volatility_of_volatility", {0.005, pg::BumpScale::relative}},
        {"product.participation_rate", {0.005, pg::BumpScale::relative}},
        {"product.local_floor", {0.002, pg::BumpScale::absolute}},
        {"product.local_cap", {0.002, pg::BumpScale::absolute}},
        {"product.global_floor", {0.002, pg::BumpScale::absolute}},
        {"product.global_cap", {0.002, pg::BumpScale::absolute}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false};
        recipe.reconstructible_sensitivities = {{"model.spot", "0", "0"}};
        const auto models = model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_cliquets(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "schobel_zhu", "cliquet", ""}, 11668827854287667200ULL,
            models, products, model::equity::schobel_zhu::prepare_schobel_zhu_cliquet_price_gradients,
            model::equity::schobel_zhu::launch_schobel_zhu_cliquet_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::schobel_zhu::prepare_cliquet_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
