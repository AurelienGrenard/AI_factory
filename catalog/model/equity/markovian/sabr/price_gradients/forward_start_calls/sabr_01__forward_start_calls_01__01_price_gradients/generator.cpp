// Generated sabr forward_start_option selected-gradient recipe.
#include "model/equity/markovian/sabr/product/forward_start_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/markovian/sabr/price_gradients/forward_start_calls/sabr_01__forward_start_calls_01__01_price_gradients.json", "catalog/model/equity/markovian/sabr/price_gradients/forward_start_calls/sabr_01__forward_start_calls_01__01_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_gradients/forward_start_calls/sabr_01__forward_start_calls_01__01_price_gradients.json", "catalog/model/equity/markovian/sabr/prices/forward_start_calls/sabr_01__forward_start_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.moneyness", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::sabr::load_models(recipe.model_input);
        const auto products = product::load_forward_start_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "forward_start_option", ""}, 11668827751208452096ULL,
            models, products, model::equity::sabr::prepare_sabr_forward_start_option_price_gradients,
            model::equity::sabr::launch_sabr_forward_start_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::sabr::prepare_forward_start_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
