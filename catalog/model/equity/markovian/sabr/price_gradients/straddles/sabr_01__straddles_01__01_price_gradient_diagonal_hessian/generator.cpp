// Generated sabr straddle gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/sabr/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/sabr/price_gradients/straddles/sabr_01__straddles_01__01_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/sabr/price_gradients/straddles/sabr_01__straddles_01__01_price_gradient_diagonal_hessian/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_gradients/straddles/sabr_01__straddles_01__01_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/sabr/prices/straddles/sabr_01__straddles_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::sabr::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::sabr::prepare_sabr_straddle_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "straddle", ""},
            11668827794158125056ULL, models, products, prepare,
            model::equity::sabr::launch_sabr_straddle_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            model::equity::sabr::sabr_straddle_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>,
            model::equity::sabr::launch_sabr_straddle_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::sabr::prepare_straddle_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
