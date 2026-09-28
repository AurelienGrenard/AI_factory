// Generated merton asian_option selected-gradient recipe.
#include "model/equity/markovian/merton/product/asian_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/asian_option/asian_options_01.json", "datasets/model/equity/markovian/merton/price_sensitivities/asian_puts/merton_01__asian_puts_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/merton/price_sensitivities/asian_puts/merton_01__asian_puts_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_sensitivities/asian_puts/merton_01__asian_puts_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/merton/prices/asian_puts/merton_01__asian_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_asian_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "merton", "asian_option", ""}, 11668827420495970304ULL,
            models, products, model::equity::merton::prepare_merton_asian_option_price_gradients,
            model::equity::merton::launch_merton_asian_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::merton::prepare_asian_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
