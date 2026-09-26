// Generated merton up_no_touch gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/merton/product/up_no_touch_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/up_no_touch/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/up_no_touch/up_no_touches_01.json", "datasets/model/equity/markovian/merton/price_gradients/up_no_touches/merton_01__up_no_touches_01__01_price_gradients_diagonal.json", "catalog/model/equity/markovian/merton/price_gradients/up_no_touches/merton_01__up_no_touches_01__01_price_gradients_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_gradients/up_no_touches/merton_01__up_no_touches_01__01_price_gradients_diagonal.json", "catalog/model/equity/markovian/merton/prices/up_no_touches/merton_01__up_no_touches_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_up_no_touches(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::merton::prepare_merton_up_no_touch_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "merton", "up_no_touch", ""},
            11668827532165120000ULL, models, products, prepare,
            model::equity::merton::launch_merton_up_no_touch_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            model::equity::merton::merton_up_no_touch_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>,
            model::equity::merton::launch_merton_up_no_touch_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::merton::prepare_up_no_touch_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
