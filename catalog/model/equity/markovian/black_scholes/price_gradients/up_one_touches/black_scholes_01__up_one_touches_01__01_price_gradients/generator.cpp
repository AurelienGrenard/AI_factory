// Generated black_scholes up_one_touch selected-gradient recipe.
#include "model/equity/markovian/black_scholes/product/up_one_touch_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/markovian/black_scholes/price_gradients/up_one_touches/black_scholes_01__up_one_touches_01__01_price_gradients.json", "catalog/model/equity/markovian/black_scholes/price_gradients/up_one_touches/black_scholes_01__up_one_touches_01__01_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/black_scholes/price_gradients/up_one_touches/black_scholes_01__up_one_touches_01__01_price_gradients.json", "catalog/model/equity/markovian/black_scholes/prices/up_one_touches/black_scholes_01__up_one_touches_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_up_one_touches(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "black_scholes", "up_one_touch", ""}, 11668826836380418048ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_up_one_touch_price_gradients,
            model::equity::black_scholes::launch_black_scholes_up_one_touch_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_up_one_touch_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
