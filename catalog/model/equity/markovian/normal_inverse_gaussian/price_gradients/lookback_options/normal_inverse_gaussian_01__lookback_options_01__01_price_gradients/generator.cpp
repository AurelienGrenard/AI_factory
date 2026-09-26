// Generated normal_inverse_gaussian lookback_option selected-gradient recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/lookback_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/lookback_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/lookback_option/lookback_options_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_gradients/lookback_options/normal_inverse_gaussian_01__lookback_options_01__01_price_gradients.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_gradients/lookback_options/normal_inverse_gaussian_01__lookback_options_01__01_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_gradients/lookback_options/normal_inverse_gaussian_01__lookback_options_01__01_price_gradients.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/lookback_options/normal_inverse_gaussian_01__lookback_options_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.alpha", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.delta", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_lookback_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "normal_inverse_gaussian", "lookback_option", ""}, 11668827643834269696ULL,
            models, products, model::equity::normal_inverse_gaussian::prepare_normal_inverse_gaussian_lookback_option_price_gradients,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_lookback_option_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::normal_inverse_gaussian::prepare_lookback_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
