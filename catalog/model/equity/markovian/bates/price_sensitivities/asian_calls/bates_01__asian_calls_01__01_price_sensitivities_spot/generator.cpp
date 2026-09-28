// Generated bates asian_option selected-gradient recipe.
#include "model/equity/markovian/bates/product/asian_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/asian_option/asian_options_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/asian_calls/bates_01__asian_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/price_sensitivities/asian_calls/bates_01__asian_calls_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/asian_calls/bates_01__asian_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/prices/asian_calls/bates_01__asian_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_asian_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "asian_option", ""}, 11668826643106889728ULL,
            models, products, model::equity::bates::prepare_bates_asian_option_price_gradients,
            model::equity::bates::launch_bates_asian_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_asian_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
