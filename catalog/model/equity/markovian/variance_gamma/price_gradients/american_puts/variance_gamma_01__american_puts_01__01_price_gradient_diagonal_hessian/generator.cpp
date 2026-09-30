// Generated variance_gamma American gradient and diagonal Hessian with frozen exercise.
#include "model/equity/markovian/variance_gamma/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/variance_gamma/price_gradients/american_puts/variance_gamma_01__american_puts_01__01_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/variance_gamma/price_gradients/american_puts/variance_gamma_01__american_puts_01__01_price_gradient_diagonal_hessian/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_gradients/american_puts/variance_gamma_01__american_puts_01__01_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/variance_gamma/prices/american_puts/variance_gamma_01__american_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::variance_gamma::prepare_variance_gamma_american_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "variance_gamma", "american_option", ""},
            11668828103395770368ULL, models, products, prepare,
            model::equity::variance_gamma::launch_variance_gamma_american_option_diagonal_sensitivities_cuda<
                OptionSide::put, pg::SensitivityOrders::first_and_second>,
            model::equity::variance_gamma::variance_gamma_american_option_node_graph_workspace_bytes<
                OptionSide::put, pg::SensitivityOrders::first_and_second>,
            model::equity::variance_gamma::launch_variance_gamma_american_option_node_graph_sensitivities_cuda<
                OptionSide::put, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::variance_gamma::prepare_american_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
