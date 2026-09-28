// Generated schobel_zhu phoenix_autocall selected-gradient recipe.
#include "model/equity/markovian/schobel_zhu/product/phoenix_autocall_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/schobel_zhu/price_sensitivities/phoenix_autocalls/schobel_zhu_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/schobel_zhu/price_sensitivities/phoenix_autocalls/schobel_zhu_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/price_sensitivities/phoenix_autocalls/schobel_zhu_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/schobel_zhu/prices/phoenix_autocalls/schobel_zhu_01__phoenix_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_phoenix_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "schobel_zhu", "phoenix_autocall", ""}, 11668827923007143936ULL,
            models, products, model::equity::schobel_zhu::prepare_schobel_zhu_phoenix_autocall_price_gradients,
            model::equity::schobel_zhu::launch_schobel_zhu_phoenix_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::schobel_zhu::prepare_phoenix_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
