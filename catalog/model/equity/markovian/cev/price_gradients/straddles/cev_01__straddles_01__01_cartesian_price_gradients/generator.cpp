// Generated cev straddle selected-gradient recipe.
#include "model/equity/markovian/cev/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/cev/price_gradients/straddles/cev_01__straddles_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/cev/price_gradients/straddles/cev_01__straddles_01__01_cartesian_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_gradients/straddles/cev_01__straddles_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/cev/prices/straddles/cev_01__straddles_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "straddle", ""}, 11668826960934469632ULL,
            models, products, model::equity::cev::prepare_cev_straddle_price_gradients,
            model::equity::cev::launch_cev_straddle_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_straddle_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
