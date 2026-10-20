// Generated bates up_no_touch selected-gradient recipe.
#include "model/equity/markovian/bates/product/up_no_touch_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/up_no_touch/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/up_no_touch/up_no_touches_01.json", "datasets/model/equity/markovian/bates/price_gradients/up_no_touches/bates_01__up_no_touches_01__01_price_gradient.json", "catalog/model/equity/markovian/bates/price_gradients/up_no_touches/bates_01__up_no_touches_01__01_price_gradient/generation.yaml", "catalog/model/equity/markovian/bates/prices/up_no_touches/bates_01__up_no_touches_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.032, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.kappa", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.001, pg::BumpScale::absolute}},
        {"model.gamma", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_volatility", {0.04, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false};
        recipe.reconstructible_sensitivities = {{"product.cash_payoff", "price / product.cash_payoff", "standard_error / product.cash_payoff"}};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_up_no_touches(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "up_no_touch", ""}, 11668826759071006720ULL,
            models, products, model::equity::bates::prepare_bates_up_no_touch_price_gradients,
            model::equity::bates::launch_bates_up_no_touch_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_up_no_touch_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
