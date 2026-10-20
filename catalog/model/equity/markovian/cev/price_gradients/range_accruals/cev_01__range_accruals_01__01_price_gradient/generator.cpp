// Generated cev range_accrual selected-gradient recipe.
#include "model/equity/markovian/cev/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/markovian/cev/price_gradients/range_accruals/cev_01__range_accruals_01__01_price_gradient.json", "catalog/model/equity/markovian/cev/price_gradients/range_accruals/cev_01__range_accruals_01__01_price_gradient/generation.yaml", "catalog/model/equity/markovian/cev/prices/range_accruals/cev_01__range_accruals_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false};
        recipe.reconstructible_sensitivities = {{"product.coupon_rate", "(price - exp(-model.risk_free_rate * product.maturity / 252)) / product.coupon_rate", "standard_error / product.coupon_rate"}};
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_range_accruals(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "range_accrual", ""}, 11668826956639502336ULL,
            models, products, model::equity::cev::prepare_cev_range_accrual_price_gradients,
            model::equity::cev::launch_cev_range_accrual_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_range_accrual_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
