// Generated normal_inverse_gaussian geometric_asian_option selected-gradient recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/geometric_asian_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/geometric_asian_calls/normal_inverse_gaussian_01__geometric_asian_calls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/geometric_asian_calls/normal_inverse_gaussian_01__geometric_asian_calls_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/geometric_asian_calls/normal_inverse_gaussian_01__geometric_asian_calls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/geometric_asian_calls/normal_inverse_gaussian_01__geometric_asian_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.alpha", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.delta", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_geometric_asian_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "normal_inverse_gaussian", "geometric_asian_option", ""}, 11668827635244335104ULL,
            models, products, model::equity::normal_inverse_gaussian::prepare_normal_inverse_gaussian_geometric_asian_option_price_gradients,
            model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_geometric_asian_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::normal_inverse_gaussian::prepare_geometric_asian_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
