// Generated bates digital_option selected-gradient recipe.
#include "model/equity/markovian/bates/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/digital_calls/bates_01__digital_calls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/bates/price_sensitivities/digital_calls/bates_01__digital_calls_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/digital_calls/bates_01__digital_calls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/bates/prices/digital_calls/bates_01__digital_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
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
