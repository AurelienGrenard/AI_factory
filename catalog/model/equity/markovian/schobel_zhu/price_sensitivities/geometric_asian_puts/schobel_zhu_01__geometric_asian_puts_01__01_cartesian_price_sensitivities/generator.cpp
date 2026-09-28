// Generated schobel_zhu geometric_asian_option selected-gradient recipe.
#include "model/equity/markovian/schobel_zhu/product/geometric_asian_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/markovian/schobel_zhu/price_sensitivities/geometric_asian_puts/schobel_zhu_01__geometric_asian_puts_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/schobel_zhu/price_sensitivities/geometric_asian_puts/schobel_zhu_01__geometric_asian_puts_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/price_sensitivities/geometric_asian_puts/schobel_zhu_01__geometric_asian_puts_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/schobel_zhu/prices/geometric_asian_puts/schobel_zhu_01__geometric_asian_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_geometric_asian_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "schobel_zhu", "geometric_asian_option", ""}, 11668827914417209344ULL,
            models, products, model::equity::schobel_zhu::prepare_schobel_zhu_geometric_asian_option_price_gradients,
            model::equity::schobel_zhu::launch_schobel_zhu_geometric_asian_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::schobel_zhu::prepare_geometric_asian_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
