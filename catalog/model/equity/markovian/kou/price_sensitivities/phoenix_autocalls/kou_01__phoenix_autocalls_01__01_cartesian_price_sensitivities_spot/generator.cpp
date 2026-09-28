// Generated kou phoenix_autocall selected-gradient recipe.
#include "model/equity/markovian/kou/product/phoenix_autocall_price_gradients.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/kou/parameters/kou_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/kou/price_sensitivities/phoenix_autocalls/kou_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/kou/price_sensitivities/phoenix_autocalls/kou_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/kou/price_sensitivities/phoenix_autocalls/kou_01__phoenix_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/kou/prices/phoenix_autocalls/kou_01__phoenix_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::kou::load_models(recipe.model_input);
        const auto products = product::load_phoenix_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "kou", "phoenix_autocall", ""}, 11668827364661395456ULL,
            models, products, model::equity::kou::prepare_kou_phoenix_autocall_price_gradients,
            model::equity::kou::launch_kou_phoenix_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::kou::prepare_phoenix_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
