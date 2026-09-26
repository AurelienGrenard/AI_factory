// Generated normal_inverse_gaussian athena_autocall selected-gradient recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/athena_autocall_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_gradients/athena_autocalls/normal_inverse_gaussian_01__athena_autocalls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_gradients/athena_autocalls/normal_inverse_gaussian_01__athena_autocalls_01__01_cartesian_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_gradients/athena_autocalls/normal_inverse_gaussian_01__athena_autocalls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/athena_autocalls/normal_inverse_gaussian_01__athena_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.alpha", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.delta", {0.005, pg::BumpScale::relative}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_athena_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "normal_inverse_gaussian", "athena_autocall", ""}, 11668827575114792960ULL,
            models, products, model::equity::normal_inverse_gaussian::prepare_normal_inverse_gaussian_athena_autocall_price_gradients,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_athena_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::normal_inverse_gaussian::prepare_athena_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
