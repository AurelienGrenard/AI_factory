// Generated log_modulated_rough_bergomi digital_option aligned first-order rough graph recipe.
#include "model/equity/rough/log_modulated_rough_bergomi/product/digital_option_price_gradients.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/digital_option.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/price_gradients/rough_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace binding = model::equity::log_modulated_rough_bergomi;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_gradients/digital_puts/log_modulated_rough_bergomi_01__digital_puts_01__01_price_gradient.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_gradients/digital_puts/log_modulated_rough_bergomi_01__digital_puts_01__01_price_gradient/generation.yaml",
            "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/digital_puts/log_modulated_rough_bergomi_01__digital_puts_01__01/recipe.yaml", PriceConstruction::Aligned, {{
        {"model.spot", {0.01, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.xi_0", {0.002, pg::BumpScale::absolute}},
        {"model.eta", {0.01, pg::BumpScale::relative}},
        {"model.hurst_exponent", {0.004, pg::BumpScale::absolute}},
        {"model.rho", {0.008, pg::BumpScale::absolute}},
        {"model.log_modulation_scale", {0.02, pg::BumpScale::relative}},
        {"model.log_modulation_power", {0.01, pg::BumpScale::relative}},
        {"product.strike", {0.01, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first, {},
            pg::SensitivityStrategy::node_graph
        };
        recipe.option_side = "put";
        recipe.reconstructible_sensitivities = {{"product.cash_payoff", "price / product.cash_payoff", "standard_error / product.cash_payoff"}};
        const auto models = binding::load_models(recipe.model_input);
        const auto products = product::load_digital_options(recipe.product_input);
        using Graph = binding::DigitalOptionNodeGraph<
            OptionSide::put, pg::SensitivityOrders::first, 16U>;
        const auto prepare = [](const auto& model_rows,
                                const auto& product_rows,
                                PriceConstruction construction,
                                pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return binding::prepare_log_modulated_rough_bergomi_digital_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first});
        };
        return offline::pricing::price_gradients::execute_rough_graph_dataset<Graph>(
            recipe,
            {offline::cuda_tuning::PricingFamily::rough_fft,
             "log_modulated_rough_bergomi", "digital_option", ""},
            11668828270899494912ULL, models, products, prepare,
            offline::cuda_tuning::kProductionPathsPerPrice);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
