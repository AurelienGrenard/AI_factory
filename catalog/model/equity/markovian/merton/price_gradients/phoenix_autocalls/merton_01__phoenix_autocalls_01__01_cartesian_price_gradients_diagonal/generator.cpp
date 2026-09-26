// Generated merton phoenix_autocall gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/merton/product/phoenix_autocall_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/merton/price_gradients/phoenix_autocalls/merton_01__phoenix_autocalls_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/equity/markovian/merton/price_gradients/phoenix_autocalls/merton_01__phoenix_autocalls_01__01_cartesian_price_gradients_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_gradients/phoenix_autocalls/merton_01__phoenix_autocalls_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/equity/markovian/merton/prices/phoenix_autocalls/merton_01__phoenix_autocalls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.coupon_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_phoenix_autocalls(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::merton::prepare_merton_phoenix_autocall_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_exact_mc, "merton", "phoenix_autocall", ""},
            11668827506395316224ULL, models, products, prepare,
            model::equity::merton::launch_merton_phoenix_autocall_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            model::equity::merton::merton_phoenix_autocall_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>,
            model::equity::merton::launch_merton_phoenix_autocall_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::merton::prepare_phoenix_autocall_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
