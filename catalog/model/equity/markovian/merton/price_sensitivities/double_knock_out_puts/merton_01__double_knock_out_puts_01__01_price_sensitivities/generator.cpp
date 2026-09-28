// Generated merton double_knock_out_option selected-gradient recipe.
#include "model/equity/markovian/merton/product/double_knock_out_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/double_knock_out_option/double_knock_out_options_01.json", "datasets/model/equity/markovian/merton/price_sensitivities/double_knock_out_puts/merton_01__double_knock_out_puts_01__01_price_sensitivities.json", "catalog/model/equity/markovian/merton/price_sensitivities/double_knock_out_puts/merton_01__double_knock_out_puts_01__01_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_sensitivities/double_knock_out_puts/merton_01__double_knock_out_puts_01__01_price_sensitivities.json", "catalog/model/equity/markovian/merton/prices/double_knock_out_puts/merton_01__double_knock_out_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_double_knock_out_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "merton", "double_knock_out_option", ""}, 11668827454855708672ULL,
            models, products, model::equity::merton::prepare_merton_double_knock_out_option_price_gradients,
            model::equity::merton::launch_merton_double_knock_out_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::merton::prepare_double_knock_out_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
