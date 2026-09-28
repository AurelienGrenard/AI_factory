// Generated sabr athena_autocall gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/sabr/product/athena_autocall_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/markovian/sabr/price_sensitivities/athena_autocalls/sabr_01__athena_autocalls_01__01_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/sabr/price_sensitivities/athena_autocalls/sabr_01__athena_autocalls_01__01_price_sensitivities_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_sensitivities/athena_autocalls/sabr_01__athena_autocalls_01__01_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/sabr/prices/athena_autocalls/sabr_01__athena_autocalls_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::sabr::load_models(recipe.model_input);
        const auto products = product::load_athena_autocalls(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::sabr::prepare_sabr_athena_autocall_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "athena_autocall", ""},
            11668827708258779136ULL, models, products, prepare,
            model::equity::sabr::launch_sabr_athena_autocall_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            model::equity::sabr::sabr_athena_autocall_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>,
            model::equity::sabr::launch_sabr_athena_autocall_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::sabr::prepare_athena_autocall_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
