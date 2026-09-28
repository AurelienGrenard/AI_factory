// Generated heston_3_2 phoenix_memory_autocall selected-gradient recipe.
#include "model/equity/markovian/heston_3_2/product/phoenix_memory_autocall_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json", "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json", "datasets/model/equity/markovian/heston_3_2/price_sensitivities/phoenix_memory_autocalls/heston_3_2_01__phoenix_memory_autocalls_01__01_price_sensitivities.json", "catalog/model/equity/markovian/heston_3_2/price_sensitivities/phoenix_memory_autocalls/heston_3_2_01__phoenix_memory_autocalls_01__01_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/heston_3_2/price_sensitivities/phoenix_memory_autocalls/heston_3_2_01__phoenix_memory_autocalls_01__01_price_sensitivities.json", "catalog/model/equity/markovian/heston_3_2/prices/phoenix_memory_autocalls/heston_3_2_01__phoenix_memory_autocalls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.coupon_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::heston_3_2::load_models(recipe.model_input);
        const auto products = product::load_phoenix_memory_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "heston_3_2", "phoenix_memory_autocall", ""}, 11668827227222441984ULL,
            models, products, model::equity::heston_3_2::prepare_heston_3_2_phoenix_memory_autocall_price_gradients,
            model::equity::heston_3_2::launch_heston_3_2_phoenix_memory_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::heston_3_2::prepare_phoenix_memory_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
