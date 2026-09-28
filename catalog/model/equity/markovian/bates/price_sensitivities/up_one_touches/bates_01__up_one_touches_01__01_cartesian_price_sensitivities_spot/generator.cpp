// Generated bates up_one_touch selected-gradient recipe.
#include "model/equity/markovian/bates/product/up_one_touch_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/up_one_touches/bates_01__up_one_touches_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/price_sensitivities/up_one_touches/bates_01__up_one_touches_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/up_one_touches/bates_01__up_one_touches_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/prices/up_one_touches/bates_01__up_one_touches_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_up_one_touches(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "up_one_touch", ""}, 11668826763365974016ULL,
            models, products, model::equity::bates::prepare_bates_up_one_touch_price_gradients,
            model::equity::bates::launch_bates_up_one_touch_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_up_one_touch_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
