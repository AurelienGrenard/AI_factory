// Generated bates digital_option selected-gradient recipe.
#include "model/equity/markovian/bates/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/markovian/bates/price_gradients/digital_calls/bates_01__digital_calls_01__01_price_gradient.json", "catalog/model/equity/markovian/bates/price_gradients/digital_calls/bates_01__digital_calls_01__01_price_gradient/generation.yaml", "catalog/model/equity/markovian/bates/prices/digital_calls/bates_01__digital_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.016, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.kappa", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.001, pg::BumpScale::absolute}},
        {"model.gamma", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_volatility", {0.04, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false};
        recipe.option_side = "call";
        recipe.reconstructible_sensitivities = {{"product.cash_payoff", "price / product.cash_payoff", "standard_error / product.cash_payoff"}};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_digital_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "digital_option", ""}, 11668826668876693504ULL,
            models, products, model::equity::bates::prepare_bates_digital_option_price_gradients,
            model::equity::bates::launch_bates_digital_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_digital_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
