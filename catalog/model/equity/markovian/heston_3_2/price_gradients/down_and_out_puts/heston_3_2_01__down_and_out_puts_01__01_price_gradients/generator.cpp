// Generated heston_3_2 down_and_out_option selected-gradient recipe.
#include "model/equity/markovian/heston_3_2/product/down_and_out_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "product/down_and_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json", "datasets/product/down_and_out_option/down_and_out_options_01.json", "datasets/model/equity/markovian/heston_3_2/price_gradients/down_and_out_puts/heston_3_2_01__down_and_out_puts_01__01_price_gradients.json", "catalog/model/equity/markovian/heston_3_2/price_gradients/down_and_out_puts/heston_3_2_01__down_and_out_puts_01__01_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/heston_3_2/price_gradients/down_and_out_puts/heston_3_2_01__down_and_out_puts_01__01_price_gradients.json", "catalog/model/equity/markovian/heston_3_2/prices/down_and_out_puts/heston_3_2_01__down_and_out_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::heston_3_2::load_models(recipe.model_input);
        const auto products = product::load_down_and_out_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "heston_3_2", "down_and_out_option", ""}, 11668827179977801728ULL,
            models, products, model::equity::heston_3_2::prepare_heston_3_2_down_and_out_option_price_gradients,
            model::equity::heston_3_2::launch_heston_3_2_down_and_out_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::heston_3_2::prepare_down_and_out_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
