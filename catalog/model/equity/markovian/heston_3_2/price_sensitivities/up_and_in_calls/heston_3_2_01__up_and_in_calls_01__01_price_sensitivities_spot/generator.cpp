// Generated heston_3_2 up_and_in_option selected-gradient recipe.
#include "model/equity/markovian/heston_3_2/product/up_and_in_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "product/up_and_in_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json", "datasets/product/up_and_in_option/up_and_in_options_01.json", "datasets/model/equity/markovian/heston_3_2/price_sensitivities/up_and_in_calls/heston_3_2_01__up_and_in_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/heston_3_2/price_sensitivities/up_and_in_calls/heston_3_2_01__up_and_in_calls_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/heston_3_2/price_sensitivities/up_and_in_calls/heston_3_2_01__up_and_in_calls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/heston_3_2/prices/up_and_in_calls/heston_3_2_01__up_and_in_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::heston_3_2::load_models(recipe.model_input);
        const auto products = product::load_up_and_in_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "heston_3_2", "up_and_in_option", ""}, 11668827240107343872ULL,
            models, products, model::equity::heston_3_2::prepare_heston_3_2_up_and_in_option_price_gradients,
            model::equity::heston_3_2::launch_heston_3_2_up_and_in_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::heston_3_2::prepare_up_and_in_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
