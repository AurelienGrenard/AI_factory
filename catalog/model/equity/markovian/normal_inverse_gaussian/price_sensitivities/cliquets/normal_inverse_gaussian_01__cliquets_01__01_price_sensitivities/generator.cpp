// Generated normal_inverse_gaussian cliquet selected-gradient recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/cliquet_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/cliquets/normal_inverse_gaussian_01__cliquets_01__01_price_sensitivities.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/cliquets/normal_inverse_gaussian_01__cliquets_01__01_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/cliquets/normal_inverse_gaussian_01__cliquets_01__01_price_sensitivities.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/cliquets/normal_inverse_gaussian_01__cliquets_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.alpha", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.delta", {0.005, pg::BumpScale::relative}},
        {"product.participation_rate", {0.005, pg::BumpScale::relative}},
        {"product.local_floor", {0.002, pg::BumpScale::absolute}},
        {"product.local_cap", {0.002, pg::BumpScale::absolute}},
        {"product.global_floor", {0.002, pg::BumpScale::absolute}},
        {"product.global_cap", {0.002, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_cliquets(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "normal_inverse_gaussian", "cliquet", ""}, 11668827579409760256ULL,
            models, products, model::equity::normal_inverse_gaussian::prepare_normal_inverse_gaussian_cliquet_price_gradients,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_cliquet_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::normal_inverse_gaussian::prepare_cliquet_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
