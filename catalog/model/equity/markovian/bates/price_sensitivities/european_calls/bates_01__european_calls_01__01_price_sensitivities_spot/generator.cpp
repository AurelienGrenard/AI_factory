// Generated bates european_option selected-gradient recipe.
#include "model/equity/markovian/bates/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/european_calls/bates_01__european_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/price_sensitivities/european_calls/bates_01__european_calls_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/european_calls/bates_01__european_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/prices/european_calls/bates_01__european_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_european_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "european_option", ""}, 11668826694646497280ULL,
            models, products, model::equity::bates::prepare_bates_european_option_price_gradients,
            model::equity::bates::launch_bates_european_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_european_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
