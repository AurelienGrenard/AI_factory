// Generated cev up_one_touch gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/cev/product/up_one_touch_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/markovian/cev/price_sensitivities/up_one_touches/cev_01__up_one_touches_01__01_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/cev/price_sensitivities/up_one_touches/cev_01__up_one_touches_01__01_price_sensitivities_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_sensitivities/up_one_touches/cev_01__up_one_touches_01__01_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/cev/prices/up_one_touches/cev_01__up_one_touches_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_up_one_touches(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::cev::prepare_cev_up_one_touch_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "up_one_touch", ""},
            11668826978114338816ULL, models, products, prepare,
            model::equity::cev::launch_cev_up_one_touch_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            model::equity::cev::cev_up_one_touch_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>,
            model::equity::cev::launch_cev_up_one_touch_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_up_one_touch_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
