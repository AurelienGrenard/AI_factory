// Generated merton European selected-gradient recipe using the shared native runner.
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/markovian/merton/price_gradients/european_calls/merton_01__european_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/merton/price_gradients/european_calls/merton_01__european_calls_01__01_cartesian_price_gradients/dataset.yaml", "https://datasets.ai-factory.example/v1/model/equity/markovian/merton/price_gradients/european_calls/merton_01__european_calls_01__01_cartesian_price_gradients.json", "catalog/model/equity/markovian/merton/prices/european_calls/merton_01__european_calls_01__01_cartesian/generator.cpp",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_european_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "merton", "european_option", ""}, 11668827467740610560ULL,
            models, products, model::equity::merton::prepare_merton_european_option_price_gradients,
            model::equity::merton::launch_merton_european_option_price_gradients_cuda<OptionSide::call>);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
