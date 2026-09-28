// Generated normal_inverse_gaussian straddle selected-gradient recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/straddles/normal_inverse_gaussian_01__straddles_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/straddles/normal_inverse_gaussian_01__straddles_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/straddles/normal_inverse_gaussian_01__straddles_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/straddles/normal_inverse_gaussian_01__straddles_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "normal_inverse_gaussian", "straddle", ""}, 11668827661014138880ULL,
            models, products, model::equity::normal_inverse_gaussian::prepare_normal_inverse_gaussian_straddle_price_gradients,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_straddle_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::normal_inverse_gaussian::prepare_straddle_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
