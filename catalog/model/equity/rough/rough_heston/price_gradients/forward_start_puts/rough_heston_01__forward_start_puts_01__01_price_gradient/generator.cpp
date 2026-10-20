// Generated rough_heston forward_start_option aligned first-order rough graph recipe.
#include "model/equity/rough/rough_heston/product/forward_start_option_price_gradients.cuh"
#include "model/equity/rough/rough_heston/product/forward_start_option.cuh"
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::rough_heston;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/rough/rough_heston/price_gradients/forward_start_puts/rough_heston_01__forward_start_puts_01__01_price_gradient.json", "catalog/model/equity/rough/rough_heston/price_gradients/forward_start_puts/rough_heston_01__forward_start_puts_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/rough_heston/prices/forward_start_puts/rough_heston_01__forward_start_puts_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.initial_variance", {0.002, pg::BumpScale::absolute}},
        {"model.mean_reversion", {0.01, pg::BumpScale::relative}},
        {"model.variance_drift", {0.002, pg::BumpScale::absolute}},
        {"model.volatility_of_variance", {0.005, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.moneyness", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.option_side = "put";
        recipe.reconstructible_sensitivities = {{"model.spot", "price / model.spot", "standard_error / model.spot"}};
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_forward_start_options(recipe.product_input);
        using Graph = binding::ForwardStartOptionNodeGraph<
            OptionSide::put, 7U, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_rough_heston_forward_start_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_n_factor,
             "rough_heston", "forward_start_option", ""},
            11668828704691191808ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
