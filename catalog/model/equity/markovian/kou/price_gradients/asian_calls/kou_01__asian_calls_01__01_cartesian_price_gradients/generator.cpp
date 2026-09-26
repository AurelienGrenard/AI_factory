// Generated kou asian_option selected-gradient recipe.
#include "model/equity/markovian/kou/product/asian_option_price_gradients.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "product/asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/kou/parameters/kou_01.json", "datasets/product/asian_option/asian_options_01.json", "datasets/model/equity/markovian/kou/price_gradients/asian_calls/kou_01__asian_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/kou/price_gradients/asian_calls/kou_01__asian_calls_01__01_cartesian_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/kou/price_gradients/asian_calls/kou_01__asian_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/kou/prices/asian_calls/kou_01__asian_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.up_probability", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::kou::load_models(recipe.model_input);
        const auto products = product::load_asian_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "kou", "asian_option", ""}, 11668827274467082240ULL,
            models, products, model::equity::kou::prepare_kou_asian_option_price_gradients,
            model::equity::kou::launch_kou_asian_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::kou::prepare_asian_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
