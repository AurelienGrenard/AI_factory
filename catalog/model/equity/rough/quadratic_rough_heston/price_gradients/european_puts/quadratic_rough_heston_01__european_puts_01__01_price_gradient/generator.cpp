// Generated quadratic_rough_heston european_option aligned first-order rough graph recipe.
#include "model/equity/rough/quadratic_rough_heston/product/european_option_price_gradients.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/european_option.cuh"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::quadratic_rough_heston;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/rough/quadratic_rough_heston/price_gradients/european_puts/quadratic_rough_heston_01__european_puts_01__01_price_gradient.json", "catalog/model/equity/rough/quadratic_rough_heston/price_gradients/european_puts/quadratic_rough_heston_01__european_puts_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/quadratic_rough_heston/prices/european_puts/quadratic_rough_heston_01__european_puts_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.initial_feedback", {0.002, pg::BumpScale::absolute}},
        {"model.quadratic_scale", {0.005, pg::BumpScale::relative}},
        {"model.quadratic_shift", {0.002, pg::BumpScale::absolute}},
        {"model.variance_floor", {0.12617915378767733, pg::BumpScale::relative}},
        {"model.feedback_rate", {0.005, pg::BumpScale::relative}},
        {"model.feedback_volatility", {0.005, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.001, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.option_side = "put";
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_european_options(recipe.product_input);
        using Graph = binding::EuropeanOptionNodeGraph<
            OptionSide::put, 7U, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_quadratic_rough_heston_european_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_n_factor,
             "quadratic_rough_heston", "european_option", ""},
            11668828429813284864ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
