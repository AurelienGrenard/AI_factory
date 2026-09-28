// Generated cev straddle gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/cev/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/cev/price_sensitivities/straddles/cev_01__straddles_01__01_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/cev/price_sensitivities/straddles/cev_01__straddles_01__01_price_sensitivities_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_sensitivities/straddles/cev_01__straddles_01__01_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/cev/prices/straddles/cev_01__straddles_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::cev::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::cev::prepare_cev_straddle_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "straddle", ""},
            11668826960934469632ULL, models, products, prepare,
            model::equity::cev::launch_cev_straddle_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            model::equity::cev::cev_straddle_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>,
            model::equity::cev::launch_cev_straddle_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::cev::prepare_straddle_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
