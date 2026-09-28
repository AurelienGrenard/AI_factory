// Generated normal_inverse_gaussian phoenix_memory_autocall selected-gradient recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/phoenix_memory_autocall_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/phoenix_memory_autocalls/normal_inverse_gaussian_01__phoenix_memory_autocalls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/phoenix_memory_autocalls/normal_inverse_gaussian_01__phoenix_memory_autocalls_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/phoenix_memory_autocalls/normal_inverse_gaussian_01__phoenix_memory_autocalls_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/phoenix_memory_autocalls/normal_inverse_gaussian_01__phoenix_memory_autocalls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_phoenix_memory_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "normal_inverse_gaussian", "phoenix_memory_autocall", ""}, 11668827652424204288ULL,
            models, products, model::equity::normal_inverse_gaussian::prepare_normal_inverse_gaussian_phoenix_memory_autocall_price_gradients,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_phoenix_memory_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::normal_inverse_gaussian::prepare_phoenix_memory_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
