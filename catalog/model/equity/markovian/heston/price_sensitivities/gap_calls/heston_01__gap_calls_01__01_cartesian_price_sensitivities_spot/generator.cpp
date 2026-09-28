// Generated heston gap_option selected-gradient recipe.
#include "model/equity/markovian/heston/product/gap_option_price_gradients.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston/parameters/heston_01.json", "datasets/product/gap_option/gap_call_options_01.json", "datasets/model/equity/markovian/heston/price_sensitivities/gap_calls/heston_01__gap_calls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/heston/price_sensitivities/gap_calls/heston_01__gap_calls_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/heston/price_sensitivities/gap_calls/heston_01__gap_calls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/heston/prices/gap_calls/heston_01__gap_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::heston::load_models(recipe.model_input);
        const auto products = product::load_gap_options(recipe.product_input, OptionSide::call);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "heston", "gap_option", ""}, 11668827068308652032ULL,
            models, products, model::equity::heston::prepare_heston_gap_option_price_gradients,
            model::equity::heston::launch_heston_gap_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::heston::prepare_gap_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
