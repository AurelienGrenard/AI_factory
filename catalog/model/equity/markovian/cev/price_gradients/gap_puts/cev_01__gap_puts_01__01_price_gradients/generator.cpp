// Generated cev gap_option selected-gradient recipe.
#include "model/equity/markovian/cev/product/gap_option_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/gap_option/gap_put_options_01.json", "datasets/model/equity/markovian/cev/price_gradients/gap_puts/cev_01__gap_puts_01__01_price_gradients.json", "catalog/model/equity/markovian/cev/price_gradients/gap_puts/cev_01__gap_puts_01__01_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_gradients/gap_puts/cev_01__gap_puts_01__01_price_gradients.json", "catalog/model/equity/markovian/cev/prices/gap_puts/cev_01__gap_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.trigger_strike", {0.005, pg::BumpScale::relative}},
        {"product.payoff_strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_gap_options(recipe.product_input, OptionSide::put);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "gap_option", ""}, 11668826930869698560ULL,
            models, products, model::equity::cev::prepare_cev_gap_option_price_gradients,
            model::equity::cev::launch_cev_gap_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_gap_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
