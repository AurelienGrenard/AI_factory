// Generated variance_gamma double_knock_out_option selected-gradient recipe.
#include "model/equity/markovian/variance_gamma/product/double_knock_out_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/double_knock_out_option/double_knock_out_options_01.json", "datasets/model/equity/markovian/variance_gamma/price_sensitivities/double_knock_out_puts/variance_gamma_01__double_knock_out_puts_01__01_price_sensitivities.json", "catalog/model/equity/markovian/variance_gamma/price_sensitivities/double_knock_out_puts/variance_gamma_01__double_knock_out_puts_01__01_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_sensitivities/double_knock_out_puts/variance_gamma_01__double_knock_out_puts_01__01_price_sensitivities.json", "catalog/model/equity/markovian/variance_gamma/prices/double_knock_out_puts/variance_gamma_01__double_knock_out_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_double_knock_out_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "variance_gamma", "double_knock_out_option", ""}, 11668828146345443328ULL,
            models, products, model::equity::variance_gamma::prepare_variance_gamma_double_knock_out_option_price_gradients,
            model::equity::variance_gamma::launch_variance_gamma_double_knock_out_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_double_knock_out_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
