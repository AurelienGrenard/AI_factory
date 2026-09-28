// Generated black_scholes asset_or_nothing_option selected-gradient recipe.
#include "model/equity/markovian/black_scholes/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/markovian/black_scholes/price_sensitivities/asset_or_nothing_calls/black_scholes_01__asset_or_nothing_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/black_scholes/price_sensitivities/asset_or_nothing_calls/black_scholes_01__asset_or_nothing_calls_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_sensitivities/asset_or_nothing_calls/black_scholes_01__asset_or_nothing_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/black_scholes/prices/asset_or_nothing_calls/black_scholes_01__asset_or_nothing_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_asset_or_nothing_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<false>(recipe,
            {offline::cuda_tuning::PricingFamily::closed_form, "black_scholes", "asset_or_nothing_option", ""}, 0ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_asset_or_nothing_option_price_gradients,
            model::equity::black_scholes::launch_black_scholes_asset_or_nothing_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_asset_or_nothing_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
