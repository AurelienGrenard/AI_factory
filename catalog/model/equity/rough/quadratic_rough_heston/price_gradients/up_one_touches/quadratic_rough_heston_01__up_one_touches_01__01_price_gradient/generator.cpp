// Generated quadratic_rough_heston up_one_touch aligned first-order rough graph recipe.
#include "model/equity/rough/quadratic_rough_heston/product/up_one_touch_price_gradients.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/up_one_touch.cuh"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::quadratic_rough_heston;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/rough/quadratic_rough_heston/price_gradients/up_one_touches/quadratic_rough_heston_01__up_one_touches_01__01_price_gradient.json", "catalog/model/equity/rough/quadratic_rough_heston/price_gradients/up_one_touches/quadratic_rough_heston_01__up_one_touches_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/quadratic_rough_heston/prices/up_one_touches/quadratic_rough_heston_01__up_one_touches_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_feedback", {0.008, pg::BumpScale::absolute}},
        {"model.quadratic_scale", {0.02, pg::BumpScale::relative}},
        {"model.quadratic_shift", {0.008, pg::BumpScale::absolute}},
        {"model.variance_floor", {0.12617915378767733, pg::BumpScale::relative}},
        {"model.feedback_rate", {0.01, pg::BumpScale::relative}},
        {"model.feedback_volatility", {0.005, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.001, pg::BumpScale::absolute}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_up_one_touches(recipe.product_input);
        using Graph = binding::UpOneTouchNodeGraph<
            7U, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_quadratic_rough_heston_up_one_touch_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_n_factor,
             "quadratic_rough_heston", "up_one_touch", ""},
            11668828494237794304ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
