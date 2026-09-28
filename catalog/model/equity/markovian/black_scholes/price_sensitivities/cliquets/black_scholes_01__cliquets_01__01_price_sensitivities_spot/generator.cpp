// Generated black_scholes cliquet selected-gradient recipe.
#include "model/equity/markovian/black_scholes/product/cliquet_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/markovian/black_scholes/price_sensitivities/cliquets/black_scholes_01__cliquets_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/black_scholes/price_sensitivities/cliquets/black_scholes_01__cliquets_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/black_scholes/price_sensitivities/cliquets/black_scholes_01__cliquets_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/black_scholes/prices/cliquets/black_scholes_01__cliquets_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_cliquets(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "black_scholes", "cliquet", ""}, 11668826789135777792ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_cliquet_price_gradients,
            model::equity::black_scholes::launch_black_scholes_cliquet_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_cliquet_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
