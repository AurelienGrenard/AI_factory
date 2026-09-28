// Generated bates phoenix_memory_autocall selected-gradient recipe.
#include "model/equity/markovian/bates/product/phoenix_memory_autocall_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/price_sensitivities/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/prices/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_phoenix_memory_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "phoenix_memory_autocall", ""}, 11668826737596170240ULL,
            models, products, model::equity::bates::prepare_bates_phoenix_memory_autocall_price_gradients,
            model::equity::bates::launch_bates_phoenix_memory_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_phoenix_memory_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
