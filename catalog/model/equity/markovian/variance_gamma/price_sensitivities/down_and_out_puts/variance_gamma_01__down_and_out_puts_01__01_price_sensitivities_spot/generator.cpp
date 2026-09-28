// Generated variance_gamma down_and_out_option selected-gradient recipe.
#include "model/equity/markovian/variance_gamma/product/down_and_out_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/down_and_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/down_and_out_option/down_and_out_options_01.json", "datasets/model/equity/markovian/variance_gamma/price_sensitivities/down_and_out_puts/variance_gamma_01__down_and_out_puts_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/variance_gamma/price_sensitivities/down_and_out_puts/variance_gamma_01__down_and_out_puts_01__01_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_sensitivities/down_and_out_puts/variance_gamma_01__down_and_out_puts_01__01_price_sensitivities_spot.json", "catalog/model/equity/markovian/variance_gamma/prices/down_and_out_puts/variance_gamma_01__down_and_out_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_down_and_out_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "variance_gamma", "down_and_out_option", ""}, 11668828154935377920ULL,
            models, products, model::equity::variance_gamma::prepare_variance_gamma_down_and_out_option_price_gradients,
            model::equity::variance_gamma::launch_variance_gamma_down_and_out_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_down_and_out_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
