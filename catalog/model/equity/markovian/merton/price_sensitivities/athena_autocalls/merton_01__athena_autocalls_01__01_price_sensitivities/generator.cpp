// Generated merton athena_autocall selected-gradient recipe.
#include "model/equity/markovian/merton/product/athena_autocall_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/markovian/merton/price_sensitivities/athena_autocalls/merton_01__athena_autocalls_01__01_price_sensitivities.json", "catalog/model/equity/markovian/merton/price_sensitivities/athena_autocalls/merton_01__athena_autocalls_01__01_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_sensitivities/athena_autocalls/merton_01__athena_autocalls_01__01_price_sensitivities.json", "catalog/model/equity/markovian/merton/prices/athena_autocalls/merton_01__athena_autocalls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_athena_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "merton", "athena_autocall", ""}, 11668827433380872192ULL,
            models, products, model::equity::merton::prepare_merton_athena_autocall_price_gradients,
            model::equity::merton::launch_merton_athena_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::merton::prepare_athena_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
