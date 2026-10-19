// Generated rough_sabr up_and_in_option aligned first-order rough graph recipe.
#include "model/equity/rough/rough_sabr/product/up_and_in_option_price_gradients.cuh"
#include "model/equity/rough/rough_sabr/product/up_and_in_option.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/up_and_in_option/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::rough_sabr;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/up_and_in_option/up_and_in_options_01.json", "datasets/model/equity/rough/rough_sabr/price_gradients/up_and_in_calls/rough_sabr_01__up_and_in_calls_01__01_price_gradient.json", "catalog/model/equity/rough/rough_sabr/price_gradients/up_and_in_calls/rough_sabr_01__up_and_in_calls_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/rough_sabr/prices/up_and_in_calls/rough_sabr_01__up_and_in_calls_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.xi_0", {0.001, pg::BumpScale::absolute}},
        {"model.eta", {0.005, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.option_side = "call";
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_up_and_in_options(recipe.product_input);
        using Graph = binding::UpAndInOptionNodeGraph<
            OptionSide::call, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_rough_sabr_up_and_in_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_fft,
             "rough_sabr", "up_and_in_option", ""},
            11668828880784850944ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
