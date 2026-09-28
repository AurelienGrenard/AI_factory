// Generated kou American selected-gradient recipe with frozen exercise.
#include "model/equity/markovian/kou/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/kou/parameters/kou_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/kou/price_sensitivities/american_calls/kou_01__american_calls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/kou/price_sensitivities/american_calls/kou_01__american_calls_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/kou/price_sensitivities/american_calls/kou_01__american_calls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/kou/prices/american_calls/kou_01__american_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::kou::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "kou", "american_option", ""}, 11668827265877147648ULL,
            models, products, model::equity::kou::prepare_kou_american_option_price_gradients,
            model::equity::kou::launch_kou_american_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::kou::prepare_american_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
