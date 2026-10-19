// Generated rough_heston down_and_out_option aligned first-order rough graph recipe.
#include "model/equity/rough/rough_heston/product/down_and_out_option_price_gradients.cuh"
#include "model/equity/rough/rough_heston/product/down_and_out_option.cuh"
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "product/down_and_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::rough_heston;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json", "datasets/product/down_and_out_option/down_and_out_options_01.json", "datasets/model/equity/rough/rough_heston/price_gradients/down_and_out_puts/rough_heston_01__down_and_out_puts_01__01_price_gradient.json", "catalog/model/equity/rough/rough_heston/price_gradients/down_and_out_puts/rough_heston_01__down_and_out_puts_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/rough_heston/prices/down_and_out_puts/rough_heston_01__down_and_out_puts_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.01, pg::BumpScale::relative}},
        {"model.initial_variance", {0.002, pg::BumpScale::absolute}},
        {"model.mean_reversion", {0.01, pg::BumpScale::relative}},
        {"model.variance_drift", {0.002, pg::BumpScale::absolute}},
        {"model.volatility_of_variance", {0.005, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.002, pg::BumpScale::absolute}},
        {"model.rho", {0.004, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.01, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.option_side = "put";
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_down_and_out_options(recipe.product_input);
        using Graph = binding::DownAndOutOptionNodeGraph<
            OptionSide::put, 7U, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_rough_heston_down_and_out_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_n_factor,
             "rough_heston", "down_and_out_option", ""},
            11668828687511322624ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
