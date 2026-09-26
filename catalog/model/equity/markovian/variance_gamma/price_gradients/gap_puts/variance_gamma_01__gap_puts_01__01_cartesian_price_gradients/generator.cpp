// Generated variance_gamma gap_option selected-gradient recipe.
#include "model/equity/markovian/variance_gamma/product/gap_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/gap_option/gap_put_options_01.json", "datasets/model/equity/markovian/variance_gamma/price_gradients/gap_puts/variance_gamma_01__gap_puts_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/variance_gamma/price_gradients/gap_puts/variance_gamma_01__gap_puts_01__01_cartesian_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_gradients/gap_puts/variance_gamma_01__gap_puts_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/variance_gamma/prices/gap_puts/variance_gamma_01__gap_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.trigger_strike", {0.005, pg::BumpScale::relative}},
        {"product.payoff_strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_gap_options(recipe.product_input, OptionSide::put);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "variance_gamma", "gap_option", ""}, 11668828180705181696ULL,
            models, products, model::equity::variance_gamma::prepare_variance_gamma_gap_option_price_gradients,
            model::equity::variance_gamma::launch_variance_gamma_gap_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_gap_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
