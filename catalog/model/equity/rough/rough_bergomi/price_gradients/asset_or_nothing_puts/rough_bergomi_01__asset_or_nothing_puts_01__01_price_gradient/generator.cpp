// Generated rough_bergomi asset_or_nothing_option aligned first-order rough graph recipe.
#include "model/equity/rough/rough_bergomi/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/rough/rough_bergomi/product/asset_or_nothing_option.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::rough_bergomi;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_gradients/asset_or_nothing_puts/rough_bergomi_01__asset_or_nothing_puts_01__01_price_gradient.json", "catalog/model/equity/rough/rough_bergomi/price_gradients/asset_or_nothing_puts/rough_bergomi_01__asset_or_nothing_puts_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/rough_bergomi/prices/asset_or_nothing_puts/rough_bergomi_01__asset_or_nothing_puts_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.01, pg::BumpScale::relative}},
        {"model.xi_0", {0.002, pg::BumpScale::absolute}},
        {"model.eta", {0.01, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.002, pg::BumpScale::absolute}},
        {"model.rho", {0.004, pg::BumpScale::absolute}},
        {"product.strike", {0.01, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.option_side = "put";
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_asset_or_nothing_options(recipe.product_input);
        using Graph = binding::AssetOrNothingOptionNodeGraph<
            OptionSide::put, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_rough_bergomi_asset_or_nothing_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_fft,
             "rough_bergomi", "asset_or_nothing_option", ""},
            11668828520007598080ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
