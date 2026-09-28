// Generated sabr phoenix_autocall selected-gradient recipe.
#include "model/equity/markovian/sabr/product/phoenix_autocall_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/sabr/price_sensitivities/phoenix_autocalls/sabr_01__phoenix_autocalls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/sabr/price_sensitivities/phoenix_autocalls/sabr_01__phoenix_autocalls_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_sensitivities/phoenix_autocalls/sabr_01__phoenix_autocalls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/sabr/prices/phoenix_autocalls/sabr_01__phoenix_autocalls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::sabr::load_models(recipe.model_input);
        const auto products = product::load_phoenix_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "phoenix_autocall", ""}, 11668827781273223168ULL,
            models, products, model::equity::sabr::prepare_sabr_phoenix_autocall_price_gradients,
            model::equity::sabr::launch_sabr_phoenix_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::sabr::prepare_phoenix_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
