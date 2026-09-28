// Generated heston up_and_out_option selected-gradient recipe.
#include "model/equity/markovian/heston/product/up_and_out_option_price_gradients.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "product/up_and_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston/parameters/heston_01.json", "datasets/product/up_and_out_option/up_and_out_options_01.json", "datasets/model/equity/markovian/heston/price_sensitivities/up_and_out_calls/heston_01__up_and_out_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/heston/price_sensitivities/up_and_out_calls/heston_01__up_and_out_calls_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/heston/price_sensitivities/up_and_out_calls/heston_01__up_and_out_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/heston/prices/up_and_out_calls/heston_01__up_and_out_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::heston::load_models(recipe.model_input);
        const auto products = product::load_up_and_out_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "heston", "up_and_out_option", ""}, 11668827111258324992ULL,
            models, products, model::equity::heston::prepare_heston_up_and_out_option_price_gradients,
            model::equity::heston::launch_heston_up_and_out_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::heston::prepare_up_and_out_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
