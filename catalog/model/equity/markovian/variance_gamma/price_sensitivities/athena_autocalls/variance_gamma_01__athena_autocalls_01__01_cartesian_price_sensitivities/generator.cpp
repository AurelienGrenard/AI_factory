// Generated variance_gamma athena_autocall selected-gradient recipe.
#include "model/equity/markovian/variance_gamma/product/athena_autocall_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/markovian/variance_gamma/price_sensitivities/athena_autocalls/variance_gamma_01__athena_autocalls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/variance_gamma/price_sensitivities/athena_autocalls/variance_gamma_01__athena_autocalls_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_sensitivities/athena_autocalls/variance_gamma_01__athena_autocalls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/variance_gamma/prices/athena_autocalls/variance_gamma_01__athena_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_athena_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "variance_gamma", "athena_autocall", ""}, 11668828124870606848ULL,
            models, products, model::equity::variance_gamma::prepare_variance_gamma_athena_autocall_price_gradients,
            model::equity::variance_gamma::launch_variance_gamma_athena_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_athena_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
