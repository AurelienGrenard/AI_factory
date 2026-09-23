// Generated kou digital_option gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/kou/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/kou/parameters/kou_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/markovian/kou/price_gradients/digital_calls/kou_01__digital_calls_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/equity/markovian/kou/price_gradients/digital_calls/kou_01__digital_calls_01__01_cartesian_price_gradients_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/kou/price_gradients/digital_calls/kou_01__digital_calls_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/equity/markovian/kou/prices/digital_calls/kou_01__digital_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.up_probability", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second};
        const auto models = model::equity::kou::load_models(recipe.model_input);
        const auto products = product::load_digital_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::kou::prepare_kou_digital_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "kou", "digital_option", ""},
            11668827300236886016ULL, models, products, prepare,
            model::equity::kou::launch_kou_digital_option_diagonal_sensitivities_cuda<
                OptionSide::call, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::kou::prepare_digital_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
