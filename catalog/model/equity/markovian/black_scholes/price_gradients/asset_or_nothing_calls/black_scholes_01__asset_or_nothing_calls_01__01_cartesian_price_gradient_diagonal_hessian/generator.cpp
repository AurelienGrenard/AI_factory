// Generated black_scholes asset_or_nothing_option gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/black_scholes/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/markovian/black_scholes/price_gradients/asset_or_nothing_calls/black_scholes_01__asset_or_nothing_calls_01__01_cartesian_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/black_scholes/price_gradients/asset_or_nothing_calls/black_scholes_01__asset_or_nothing_calls_01__01_cartesian_price_gradient_diagonal_hessian/generation.yaml", "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_gradients/asset_or_nothing_calls/black_scholes_01__asset_or_nothing_calls_01__01_cartesian_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/black_scholes/prices/asset_or_nothing_calls/black_scholes_01__asset_or_nothing_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_asset_or_nothing_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::black_scholes::prepare_black_scholes_asset_or_nothing_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_dataset<
            false, pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::closed_form, "black_scholes", "asset_or_nothing_option", ""},
            0ULL, models, products, prepare,
            model::equity::black_scholes::launch_black_scholes_asset_or_nothing_option_diagonal_sensitivities_cuda<
                OptionSide::call, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_asset_or_nothing_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
