// Generated cev digital_option gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/cev/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/markovian/cev/price_gradients/digital_puts/cev_01__digital_puts_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/equity/markovian/cev/price_gradients/digital_puts/cev_01__digital_puts_01__01_cartesian_price_gradients_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_gradients/digital_puts/cev_01__digital_puts_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/equity/markovian/cev/prices/digital_puts/cev_01__digital_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_digital_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::cev::prepare_cev_digital_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "digital_option", ""},
            11668826887920025600ULL, models, products, prepare,
            model::equity::cev::launch_cev_digital_option_diagonal_sensitivities_cuda<
                OptionSide::put, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_digital_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
