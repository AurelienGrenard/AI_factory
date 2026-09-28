// Generated bates range_accrual selected-gradient recipe.
#include "model/equity/markovian/bates/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/range_accruals/bates_01__range_accruals_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/price_sensitivities/range_accruals/bates_01__range_accruals_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/range_accruals/bates_01__range_accruals_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/bates/prices/range_accruals/bates_01__range_accruals_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_range_accruals(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "range_accrual", ""}, 11668826741891137536ULL,
            models, products, model::equity::bates::prepare_bates_range_accrual_price_gradients,
            model::equity::bates::launch_bates_range_accrual_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_range_accrual_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
