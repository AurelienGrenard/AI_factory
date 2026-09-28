// Generated variance_gamma phoenix_autocall selected-gradient recipe.
#include "model/equity/markovian/variance_gamma/product/phoenix_autocall_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/variance_gamma/price_sensitivities/phoenix_autocalls/variance_gamma_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/variance_gamma/price_sensitivities/phoenix_autocalls/variance_gamma_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_sensitivities/phoenix_autocalls/variance_gamma_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/variance_gamma/prices/phoenix_autocalls/variance_gamma_01__phoenix_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_phoenix_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "variance_gamma", "phoenix_autocall", ""}, 11668828197885050880ULL,
            models, products, model::equity::variance_gamma::prepare_variance_gamma_phoenix_autocall_price_gradients,
            model::equity::variance_gamma::launch_variance_gamma_phoenix_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_phoenix_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
