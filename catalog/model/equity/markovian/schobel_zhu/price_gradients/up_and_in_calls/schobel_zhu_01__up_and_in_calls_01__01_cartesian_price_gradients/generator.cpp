// Generated schobel_zhu up_and_in_option selected-gradient recipe.
#include "model/equity/markovian/schobel_zhu/product/up_and_in_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/up_and_in_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/up_and_in_option/up_and_in_options_01.json", "datasets/model/equity/markovian/schobel_zhu/price_gradients/up_and_in_calls/schobel_zhu_01__up_and_in_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/schobel_zhu/price_gradients/up_and_in_calls/schobel_zhu_01__up_and_in_calls_01__01_cartesian_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/price_gradients/up_and_in_calls/schobel_zhu_01__up_and_in_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/schobel_zhu/prices/up_and_in_calls/schobel_zhu_01__up_and_in_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_up_and_in_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "schobel_zhu", "up_and_in_option", ""}, 11668827940187013120ULL,
            models, products, model::equity::schobel_zhu::prepare_schobel_zhu_up_and_in_option_price_gradients,
            model::equity::schobel_zhu::launch_schobel_zhu_up_and_in_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::schobel_zhu::prepare_up_and_in_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
