// Generated bates straddle selected-gradient recipe.
#include "model/equity/markovian/bates/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/straddles/bates_01__straddles_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/price_sensitivities/straddles/bates_01__straddles_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/straddles/bates_01__straddles_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/prices/straddles/bates_01__straddles_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "straddle", ""}, 11668826746186104832ULL,
            models, products, model::equity::bates::prepare_bates_straddle_price_gradients,
            model::equity::bates::launch_bates_straddle_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_straddle_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
