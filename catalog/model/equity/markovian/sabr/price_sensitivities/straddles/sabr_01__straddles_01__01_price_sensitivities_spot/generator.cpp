// Generated sabr straddle selected-gradient recipe.
#include "model/equity/markovian/sabr/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/sabr/price_sensitivities/straddles/sabr_01__straddles_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/sabr/price_sensitivities/straddles/sabr_01__straddles_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_sensitivities/straddles/sabr_01__straddles_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/sabr/prices/straddles/sabr_01__straddles_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::sabr::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "straddle", ""}, 11668827794158125056ULL,
            models, products, model::equity::sabr::prepare_sabr_straddle_price_gradients,
            model::equity::sabr::launch_sabr_straddle_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::sabr::prepare_straddle_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
