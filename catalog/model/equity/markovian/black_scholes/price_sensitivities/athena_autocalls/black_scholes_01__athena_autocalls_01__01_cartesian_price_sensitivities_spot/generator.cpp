// Generated black_scholes athena_autocall selected-gradient recipe.
#include "model/equity/markovian/black_scholes/product/athena_autocall_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/markovian/black_scholes/price_sensitivities/athena_autocalls/black_scholes_01__athena_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/black_scholes/price_sensitivities/athena_autocalls/black_scholes_01__athena_autocalls_01__01_cartesian_price_sensitivities_spot/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/black_scholes/price_sensitivities/athena_autocalls/black_scholes_01__athena_autocalls_01__01_cartesian_price_sensitivities_spot.json", "catalog/model/equity/markovian/black_scholes/prices/athena_autocalls/black_scholes_01__athena_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_athena_autocalls(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<true>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "black_scholes", "athena_autocall", ""}, 11668826784840810496ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_athena_autocall_price_gradients,
            model::equity::black_scholes::launch_black_scholes_athena_autocall_price_gradients_cuda,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_athena_autocall_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
