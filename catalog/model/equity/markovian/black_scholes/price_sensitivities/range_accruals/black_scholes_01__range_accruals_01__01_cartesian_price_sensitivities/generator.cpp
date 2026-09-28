// Generated black_scholes range_accrual selected-gradient recipe.
#include "model/equity/markovian/black_scholes/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/markovian/black_scholes/price_sensitivities/range_accruals/black_scholes_01__range_accruals_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/black_scholes/price_sensitivities/range_accruals/black_scholes_01__range_accruals_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_sensitivities/range_accruals/black_scholes_01__range_accruals_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/black_scholes/prices/range_accruals/black_scholes_01__range_accruals_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}},
        {"product.coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_range_accruals(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<false>(recipe,
            {offline::cuda_tuning::PricingFamily::closed_form, "black_scholes", "range_accrual", ""}, 0ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_range_accrual_price_gradients,
            model::equity::black_scholes::launch_black_scholes_range_accrual_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_range_accrual_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
