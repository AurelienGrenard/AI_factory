// Generated stein_stein lookback_option gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/stein_stein/product/lookback_option_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/lookback_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json", "datasets/product/lookback_option/lookback_options_01.json", "datasets/model/equity/markovian/stein_stein/price_gradients/lookback_options/stein_stein_01__lookback_options_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/equity/markovian/stein_stein/price_gradients/lookback_options/stein_stein_01__lookback_options_01__01_cartesian_price_gradients_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/stein_stein/price_gradients/lookback_options/stein_stein_01__lookback_options_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/equity/markovian/stein_stein/prices/lookback_options/stein_stein_01__lookback_options_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::stein_stein::load_models(recipe.model_input);
        const auto products = product::load_lookback_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::stein_stein::prepare_stein_stein_lookback_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "stein_stein", "lookback_option", ""},
            11668828051856162816ULL, models, products, prepare,
            model::equity::stein_stein::launch_stein_stein_lookback_option_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            model::equity::stein_stein::stein_stein_lookback_option_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>,
            model::equity::stein_stein::launch_stein_stein_lookback_option_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::stein_stein::prepare_lookback_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
