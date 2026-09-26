// Generated merton geometric_asian_option selected-gradient recipe.
#include "model/equity/markovian/merton/product/geometric_asian_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/markovian/merton/price_gradients/geometric_asian_calls/merton_01__geometric_asian_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/merton/price_gradients/geometric_asian_calls/merton_01__geometric_asian_calls_01__01_cartesian_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_gradients/geometric_asian_calls/merton_01__geometric_asian_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/merton/prices/geometric_asian_calls/merton_01__geometric_asian_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_geometric_asian_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "merton", "geometric_asian_option", ""}, 11668827493510414336ULL,
            models, products, model::equity::merton::prepare_merton_geometric_asian_option_price_gradients,
            model::equity::merton::launch_merton_geometric_asian_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::merton::prepare_geometric_asian_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
