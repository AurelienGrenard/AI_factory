// Generated cev American selected-gradient recipe with frozen exercise.
#include "model/equity/markovian/cev/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/cev/price_sensitivities/american_puts/cev_01__american_puts_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/cev/price_sensitivities/american_puts/cev_01__american_puts_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_sensitivities/american_puts/cev_01__american_puts_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/cev/prices/american_puts/cev_01__american_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "cev", "american_option", ""}, 11668826853560287232ULL,
            models, products, model::equity::cev::prepare_cev_american_option_price_gradients,
            model::equity::cev::launch_cev_american_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_american_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
