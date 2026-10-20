// Generated rough_sabr up_one_touch aligned first-order rough graph recipe.
#include "model/equity/rough/rough_sabr/product/up_one_touch_price_gradients.cuh"
#include "model/equity/rough/rough_sabr/product/up_one_touch.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::rough_sabr;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/rough/rough_sabr/price_gradients/up_one_touches/rough_sabr_01__up_one_touches_01__01_price_gradient.json", "catalog/model/equity/rough/rough_sabr/price_gradients/up_one_touches/rough_sabr_01__up_one_touches_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/rough_sabr/prices/up_one_touches/rough_sabr_01__up_one_touches_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.xi_0", {0.001, pg::BumpScale::absolute}},
        {"model.eta", {0.005, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.reconstructible_sensitivities = {{"product.cash_payoff", "price / product.cash_payoff", "standard_error / product.cash_payoff"}};
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_up_one_touches(recipe.product_input);
        using Graph = binding::UpOneTouchNodeGraph<
            pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_rough_sabr_up_one_touch_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_fft,
             "rough_sabr", "up_one_touch", ""},
            11668828893669752832ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
