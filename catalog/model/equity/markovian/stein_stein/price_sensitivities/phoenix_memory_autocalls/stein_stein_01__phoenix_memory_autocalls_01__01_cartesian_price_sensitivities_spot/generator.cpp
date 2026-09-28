// Generated stein_stein phoenix_memory_autocall selected-gradient recipe.
#include "model/equity/markovian/stein_stein/product/phoenix_memory_autocall_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json", "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json", "datasets/model/equity/markovian/stein_stein/price_sensitivities/phoenix_memory_autocalls/stein_stein_01__phoenix_memory_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/stein_stein/price_sensitivities/phoenix_memory_autocalls/stein_stein_01__phoenix_memory_autocalls_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/stein_stein/price_sensitivities/phoenix_memory_autocalls/stein_stein_01__phoenix_memory_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/stein_stein/prices/phoenix_memory_autocalls/stein_stein_01__phoenix_memory_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::stein_stein::load_models(recipe.model_input);
        const auto products = product::load_phoenix_memory_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "stein_stein", "phoenix_memory_autocall", ""}, 11668828060446097408ULL,
            models, products, model::equity::stein_stein::prepare_stein_stein_phoenix_memory_autocall_price_gradients,
            model::equity::stein_stein::launch_stein_stein_phoenix_memory_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::stein_stein::prepare_phoenix_memory_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
