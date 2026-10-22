// Generated rough_heston athena_autocall first-order rough graph recipe.
#include "model/equity/rough/rough_heston/product/athena_autocall_price_gradients.cuh"
#include "model/equity/rough/rough_heston/product/athena_autocall.cuh"
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::rough_heston;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json", "datasets/product/athena_autocall/athena_autocalls_01_shard00.json", "datasets/model/equity/rough/rough_heston/price_gradients/athena_autocalls/rough_heston_01__athena_autocalls_01__01_cartesian_shard00_price_gradient.json", "catalog/model/equity/rough/rough_heston/price_gradients/athena_autocalls/rough_heston_01__athena_autocalls_01__01_cartesian_shard00_price_gradient/generation.yaml",
            "catalog/model/equity/rough/rough_heston/prices/athena_autocalls/rough_heston_01__athena_autocalls_01__01_cartesian_shard00/recipe.yaml", PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.variance_drift", {0.001, pg::BumpScale::absolute}},
        {"model.volatility_of_variance", {0.005, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_athena_autocalls(recipe.product_input);
        using Graph = binding::AthenaAutocallNodeGraph<
            7U, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_rough_heston_athena_autocall_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_n_factor,
             "rough_heston", "athena_autocall", ""},
            11668828657446551552ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
