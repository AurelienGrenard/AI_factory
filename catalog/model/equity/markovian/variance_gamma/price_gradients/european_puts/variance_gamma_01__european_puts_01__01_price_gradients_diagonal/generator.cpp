// Generated variance_gamma european_option gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/variance_gamma/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        const datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/markovian/variance_gamma/price_gradients/european_puts/variance_gamma_01__european_puts_01__01_price_gradients_diagonal.json", "catalog/model/equity/markovian/variance_gamma/price_gradients/european_puts/variance_gamma_01__european_puts_01__01_price_gradients_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_gradients/european_puts/variance_gamma_01__european_puts_01__01_price_gradients_diagonal.json", "catalog/model/equity/markovian/variance_gamma/prices/european_puts/variance_gamma_01__european_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second};
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_european_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::variance_gamma::prepare_variance_gamma_european_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "variance_gamma", "european_option", ""},
            11668828163525312512ULL, models, products, prepare,
            model::equity::variance_gamma::launch_variance_gamma_european_option_diagonal_sensitivities_cuda<
                OptionSide::put, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_european_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
