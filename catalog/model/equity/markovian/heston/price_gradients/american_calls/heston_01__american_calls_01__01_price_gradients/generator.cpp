// Generated heston American selected-gradient recipe with frozen exercise.
#include "model/equity/markovian/heston/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston/parameters/heston_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/heston/price_gradients/american_calls/heston_01__american_calls_01__01_price_gradients.json", "catalog/model/equity/markovian/heston/price_gradients/american_calls/heston_01__american_calls_01__01_price_gradients/generation.yaml", "https://datasets.ai-factory.example/v1/model/equity/markovian/heston/price_gradients/american_calls/heston_01__american_calls_01__01_price_gradients.json", "catalog/model/equity/markovian/heston/prices/american_calls/heston_01__american_calls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        const auto models = model::equity::heston::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "heston", "american_option", ""}, 11668826990999240704ULL,
            models, products, model::equity::heston::prepare_heston_american_option_price_gradients,
            model::equity::heston::launch_heston_american_option_price_gradients_cuda<OptionSide::call>);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
