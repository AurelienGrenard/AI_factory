// Generated schobel_zhu cliquet selected-gradient recipe.
#include "model/equity/markovian/schobel_zhu/product/cliquet_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/markovian/schobel_zhu/price_sensitivities/cliquets/schobel_zhu_01__cliquets_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/schobel_zhu/price_sensitivities/cliquets/schobel_zhu_01__cliquets_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/price_sensitivities/cliquets/schobel_zhu_01__cliquets_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/schobel_zhu/prices/cliquets/schobel_zhu_01__cliquets_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"product.participation_rate", {0.005, pg::BumpScale::relative}},
        {"product.local_floor", {0.002, pg::BumpScale::absolute}},
        {"product.local_cap", {0.002, pg::BumpScale::absolute}},
        {"product.global_floor", {0.002, pg::BumpScale::absolute}},
        {"product.global_cap", {0.002, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false};
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
