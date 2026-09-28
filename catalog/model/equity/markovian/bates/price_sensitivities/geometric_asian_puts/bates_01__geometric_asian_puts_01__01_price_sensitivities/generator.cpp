// Generated bates geometric_asian_option selected-gradient recipe.
#include "model/equity/markovian/bates/product/geometric_asian_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/geometric_asian_puts/bates_01__geometric_asian_puts_01__01_price_sensitivities.json", "catalog/model/equity/markovian/bates/price_sensitivities/geometric_asian_puts/bates_01__geometric_asian_puts_01__01_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/geometric_asian_puts/bates_01__geometric_asian_puts_01__01_price_sensitivities.json", "catalog/model/equity/markovian/bates/prices/geometric_asian_puts/bates_01__geometric_asian_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_geometric_asian_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "geometric_asian_option", ""}, 11668826724711268352ULL,
            models, products, model::equity::bates::prepare_bates_geometric_asian_option_price_gradients,
            model::equity::bates::launch_bates_geometric_asian_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_geometric_asian_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
