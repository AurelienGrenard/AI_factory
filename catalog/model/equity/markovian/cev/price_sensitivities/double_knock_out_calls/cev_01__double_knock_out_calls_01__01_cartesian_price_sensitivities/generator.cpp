// Generated cev double_knock_out_option selected-gradient recipe.
#include "model/equity/markovian/cev/product/double_knock_out_option_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/double_knock_out_option/double_knock_out_options_01.json", "datasets/model/equity/markovian/cev/price_sensitivities/double_knock_out_calls/cev_01__double_knock_out_calls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/cev/price_sensitivities/double_knock_out_calls/cev_01__double_knock_out_calls_01__01_cartesian_price_sensitivities/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_sensitivities/double_knock_out_calls/cev_01__double_knock_out_calls_01__01_cartesian_price_sensitivities.json", "catalog/model/equity/markovian/cev/prices/double_knock_out_calls/cev_01__double_knock_out_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_double_knock_out_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "double_knock_out_option", ""}, 11668826892214992896ULL,
            models, products, model::equity::cev::prepare_cev_double_knock_out_option_price_gradients,
            model::equity::cev::launch_cev_double_knock_out_option_price_gradients_cuda<OptionSide::call>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_double_knock_out_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
