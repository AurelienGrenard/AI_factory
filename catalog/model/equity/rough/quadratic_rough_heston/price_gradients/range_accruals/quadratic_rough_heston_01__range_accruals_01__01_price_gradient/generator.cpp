// Generated quadratic_rough_heston range_accrual aligned first-order rough graph recipe.
#include "model/equity/rough/quadratic_rough_heston/product/range_accrual_price_gradients.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/range_accrual.cuh"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::quadratic_rough_heston;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/rough/quadratic_rough_heston/price_gradients/range_accruals/quadratic_rough_heston_01__range_accruals_01__01_price_gradient.json", "catalog/model/equity/rough/quadratic_rough_heston/price_gradients/range_accruals/quadratic_rough_heston_01__range_accruals_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/quadratic_rough_heston/prices/range_accruals/quadratic_rough_heston_01__range_accruals_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.initial_feedback", {0.004, pg::BumpScale::absolute}},
        {"model.quadratic_scale", {0.01, pg::BumpScale::relative}},
        {"model.quadratic_shift", {0.004, pg::BumpScale::absolute}},
        {"model.variance_floor", {0.12617915378767733, pg::BumpScale::relative}},
        {"model.feedback_rate", {0.01, pg::BumpScale::relative}},
        {"model.feedback_volatility", {0.01, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.002, pg::BumpScale::absolute}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.reconstructible_sensitivities = {{"product.coupon_rate", "(price - exp(-model.risk_free_rate * product.maturity / 252)) / product.coupon_rate", "standard_error / product.coupon_rate"}};
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_range_accruals(recipe.product_input);
        using Graph = binding::RangeAccrualNodeGraph<
            7U, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_quadratic_rough_heston_range_accrual_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_n_factor,
             "quadratic_rough_heston", "range_accrual", ""},
            11668828472762957824ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
