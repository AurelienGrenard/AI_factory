// Generated variance_gamma lookback_option selected-gradient recipe.
#include "model/equity/markovian/variance_gamma/product/lookback_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/lookback_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/lookback_option/lookback_options_01.json", "datasets/model/equity/markovian/variance_gamma/price_sensitivities/lookback_options/variance_gamma_01__lookback_options_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/variance_gamma/price_sensitivities/lookback_options/variance_gamma_01__lookback_options_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_sensitivities/lookback_options/variance_gamma_01__lookback_options_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/variance_gamma/prices/lookback_options/variance_gamma_01__lookback_options_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_lookback_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "variance_gamma", "lookback_option", ""}, 11668828193590083584ULL,
            models, products, model::equity::variance_gamma::prepare_variance_gamma_lookback_option_price_gradients,
            model::equity::variance_gamma::launch_variance_gamma_lookback_option_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_lookback_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
