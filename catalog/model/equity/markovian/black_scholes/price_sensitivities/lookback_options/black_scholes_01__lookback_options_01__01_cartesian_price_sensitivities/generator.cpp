// Generated black_scholes lookback_option selected-gradient recipe.
#include "model/equity/markovian/black_scholes/product/lookback_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/lookback_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/lookback_option/lookback_options_01.json", "datasets/model/equity/markovian/black_scholes/price_sensitivities/lookback_options/black_scholes_01__lookback_options_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/black_scholes/price_sensitivities/lookback_options/black_scholes_01__lookback_options_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/black_scholes/price_sensitivities/lookback_options/black_scholes_01__lookback_options_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/black_scholes/prices/lookback_options/black_scholes_01__lookback_options_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_lookback_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "black_scholes", "lookback_option", ""}, 11668826810610614272ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_lookback_option_price_gradients,
            model::equity::black_scholes::launch_black_scholes_lookback_option_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_lookback_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
