// Generated variance_gamma straddle selected-gradient recipe.
#include "model/equity/markovian/variance_gamma/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/variance_gamma/price_sensitivities/straddles/variance_gamma_01__straddles_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/variance_gamma/price_sensitivities/straddles/variance_gamma_01__straddles_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_sensitivities/straddles/variance_gamma_01__straddles_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/variance_gamma/prices/straddles/variance_gamma_01__straddles_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "variance_gamma", "straddle", ""}, 11668828210769952768ULL,
            models, products, model::equity::variance_gamma::prepare_variance_gamma_straddle_price_gradients,
            model::equity::variance_gamma::launch_variance_gamma_straddle_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_straddle_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
