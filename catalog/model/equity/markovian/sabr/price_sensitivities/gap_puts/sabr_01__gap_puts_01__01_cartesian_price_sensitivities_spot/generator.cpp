// Generated sabr gap_option selected-gradient recipe.
#include "model/equity/markovian/sabr/product/gap_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/gap_option/gap_put_options_01.json", "datasets/model/equity/markovian/sabr/price_sensitivities/gap_puts/sabr_01__gap_puts_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/sabr/price_sensitivities/gap_puts/sabr_01__gap_puts_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_sensitivities/gap_puts/sabr_01__gap_puts_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/sabr/prices/gap_puts/sabr_01__gap_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::sabr::load_models(recipe.model_input);
        const auto products = product::load_gap_options(recipe.product_input, OptionSide::put);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "gap_option", ""}, 11668827764093353984ULL,
            models, products, model::equity::sabr::prepare_sabr_gap_option_price_gradients,
            model::equity::sabr::launch_sabr_gap_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::sabr::prepare_gap_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
