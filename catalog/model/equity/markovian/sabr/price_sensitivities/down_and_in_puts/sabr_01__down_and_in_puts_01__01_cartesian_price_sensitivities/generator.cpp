// Generated sabr down_and_in_option selected-gradient recipe.
#include "model/equity/markovian/sabr/product/down_and_in_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/down_and_in_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/down_and_in_option/down_and_in_options_01.json", "datasets/model/equity/markovian/sabr/price_sensitivities/down_and_in_puts/sabr_01__down_and_in_puts_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/sabr/price_sensitivities/down_and_in_puts/sabr_01__down_and_in_puts_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_sensitivities/down_and_in_puts/sabr_01__down_and_in_puts_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/sabr/prices/down_and_in_puts/sabr_01__down_and_in_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::sabr::load_models(recipe.model_input);
        const auto products = product::load_down_and_in_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "down_and_in_option", ""}, 11668827734028582912ULL,
            models, products, model::equity::sabr::prepare_sabr_down_and_in_option_price_gradients,
            model::equity::sabr::launch_sabr_down_and_in_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::sabr::prepare_down_and_in_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
