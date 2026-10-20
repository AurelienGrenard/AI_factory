// Generated rough_bergomi european_option aligned first-order rough graph recipe.
#include "model/equity/rough/rough_bergomi/product/european_option_price_gradients.cuh"
#include "model/equity/rough/rough_bergomi/product/european_option.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::rough_bergomi;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_gradients/european_puts/rough_bergomi_01__european_puts_01__01_price_gradient.json", "catalog/model/equity/rough/rough_bergomi/price_gradients/european_puts/rough_bergomi_01__european_puts_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/rough_bergomi/prices/european_puts/rough_bergomi_01__european_puts_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.xi_0", {0.001, pg::BumpScale::absolute}},
        {"model.eta", {0.005, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
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
            OptionSide::put, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_rough_bergomi_european_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_fft,
             "rough_bergomi", "european_option", ""},
            11668828562957271040ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
