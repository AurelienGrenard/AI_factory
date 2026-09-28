// Generated cev cliquet selected-gradient recipe.
#include "model/equity/markovian/cev/product/cliquet_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/markovian/cev/price_sensitivities/cliquets/cev_01__cliquets_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/cev/price_sensitivities/cliquets/cev_01__cliquets_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_sensitivities/cliquets/cev_01__cliquets_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/cev/prices/cliquets/cev_01__cliquets_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_cliquets(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "cliquet", ""}, 11668826879330091008ULL,
            models, products, model::equity::cev::prepare_cev_cliquet_price_gradients,
            model::equity::cev::launch_cev_cliquet_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_cliquet_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
