// Generated black_scholes straddle selected-gradient recipe.
#include "model/equity/markovian/black_scholes/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/black_scholes/price_sensitivities/straddles/black_scholes_01__straddles_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/black_scholes/price_sensitivities/straddles/black_scholes_01__straddles_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_sensitivities/straddles/black_scholes_01__straddles_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/black_scholes/prices/straddles/black_scholes_01__straddles_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<false>(recipe,
            {offline::cuda_tuning::PricingFamily::closed_form, "black_scholes", "straddle", ""}, 0ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_straddle_price_gradients,
            model::equity::black_scholes::launch_black_scholes_straddle_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_straddle_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
