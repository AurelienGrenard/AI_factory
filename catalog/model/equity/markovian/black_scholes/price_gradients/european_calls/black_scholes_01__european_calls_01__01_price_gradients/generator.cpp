// Generated black_scholes European selected-gradient recipe using the shared native runner.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/markovian/black_scholes/price_gradients/european_calls/black_scholes_01__european_calls_01__01_price_gradients.json", "catalog/model/equity/markovian/black_scholes/price_gradients/european_calls/black_scholes_01__european_calls_01__01_price_gradients/dataset.yaml", "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_gradients/european_calls/black_scholes_01__european_calls_01__01_price_gradients.json", "catalog/model/equity/markovian/black_scholes/prices/european_calls/black_scholes_01__european_calls_01__01/generator.cpp",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_european_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<false>(recipe,
            {offline::cuda_tuning::PricingFamily::closed_form, "black_scholes", "european_option", ""}, 0ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_european_option_price_gradients,
            model::equity::black_scholes::launch_black_scholes_european_option_price_gradients_cuda<OptionSide::call>);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
