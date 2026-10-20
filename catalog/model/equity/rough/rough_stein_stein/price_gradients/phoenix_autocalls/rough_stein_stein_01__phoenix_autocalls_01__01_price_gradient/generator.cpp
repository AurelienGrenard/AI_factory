// Generated rough_stein_stein phoenix_autocall aligned first-order rough graph recipe.
#include "model/equity/rough/rough_stein_stein/product/phoenix_autocall_price_gradients.cuh"
#include "model/equity/rough/rough_stein_stein/product/phoenix_autocall.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::rough_stein_stein;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/rough/rough_stein_stein/price_gradients/phoenix_autocalls/rough_stein_stein_01__phoenix_autocalls_01__01_price_gradient.json", "catalog/model/equity/rough/rough_stein_stein/price_gradients/phoenix_autocalls/rough_stein_stein_01__phoenix_autocalls_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/rough_stein_stein/prices/phoenix_autocalls/rough_stein_stein_01__phoenix_autocalls_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.01, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.volatility_level", {0.01, pg::BumpScale::relative}},
        {"model.mean_reversion", {0.01, pg::BumpScale::relative}},
        {"model.volatility_of_volatility", {0.01, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.004, pg::BumpScale::absolute}},
        {"model.rho", {0.008, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.01, pg::BumpScale::relative}},
        {"product.coupon_barrier", {0.01, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.01, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_phoenix_autocalls(recipe.product_input);
        using Graph = binding::PhoenixAutocallNodeGraph<
            pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_rough_stein_stein_phoenix_autocall_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_fft,
             "rough_stein_stein", "phoenix_autocall", ""},
            11668828996748967936ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
