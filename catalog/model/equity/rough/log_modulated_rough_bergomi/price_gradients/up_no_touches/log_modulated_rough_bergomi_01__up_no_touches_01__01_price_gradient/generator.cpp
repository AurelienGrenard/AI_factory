// Generated log_modulated_rough_bergomi up_no_touch aligned first-order rough graph recipe.
#include "model/equity/rough/log_modulated_rough_bergomi/product/up_no_touch_price_gradients.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/up_no_touch.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/up_no_touch/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::log_modulated_rough_bergomi;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/up_no_touch/up_no_touches_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_gradients/up_no_touches/log_modulated_rough_bergomi_01__up_no_touches_01__01_price_gradient.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_gradients/up_no_touches/log_modulated_rough_bergomi_01__up_no_touches_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/up_no_touches/log_modulated_rough_bergomi_01__up_no_touches_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.01, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.xi_0", {0.002, pg::BumpScale::absolute}},
        {"model.eta", {0.01, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.004, pg::BumpScale::absolute}},
        {"model.rho", {0.008, pg::BumpScale::absolute}},
        {"model.log_modulation_scale", {0.02, pg::BumpScale::relative}},
        {"model.log_modulation_power", {0.01, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.reconstructible_sensitivities = {{"product.cash_payoff", "price / product.cash_payoff", "standard_error / product.cash_payoff"}};
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_up_no_touches(recipe.product_input);
        using Graph = binding::UpNoTouchNodeGraph<
            pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_log_modulated_rough_bergomi_up_no_touch_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_fft,
             "log_modulated_rough_bergomi", "up_no_touch", ""},
            11668828356798840832ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
