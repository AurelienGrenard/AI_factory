// Generated bates phoenix_autocall selected-gradient recipe.
#include "model/equity/markovian/bates/product/phoenix_autocall_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/phoenix_autocalls/bates_01__phoenix_autocalls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/bates/price_sensitivities/phoenix_autocalls/bates_01__phoenix_autocalls_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/phoenix_autocalls/bates_01__phoenix_autocalls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/bates/prices/phoenix_autocalls/bates_01__phoenix_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.coupon_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_phoenix_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "phoenix_autocall", ""}, 11668826733301202944ULL,
            models, products, model::equity::bates::prepare_bates_phoenix_autocall_price_gradients,
            model::equity::bates::launch_bates_phoenix_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_phoenix_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
