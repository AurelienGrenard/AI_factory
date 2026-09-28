// Generated schobel_zhu up_and_out_option gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/schobel_zhu/product/up_and_out_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/up_and_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/up_and_out_option/up_and_out_options_01.json", "datasets/model/equity/markovian/schobel_zhu/price_sensitivities/up_and_out_calls/schobel_zhu_01__up_and_out_calls_01__01_cartesian_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/schobel_zhu/price_sensitivities/up_and_out_calls/schobel_zhu_01__up_and_out_calls_01__01_cartesian_price_sensitivities_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/price_sensitivities/up_and_out_calls/schobel_zhu_01__up_and_out_calls_01__01_cartesian_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/schobel_zhu/prices/up_and_out_calls/schobel_zhu_01__up_and_out_calls_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_up_and_out_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::schobel_zhu::prepare_schobel_zhu_up_and_out_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "schobel_zhu", "up_and_out_option", ""},
            11668827944481980416ULL, models, products, prepare,
            model::equity::schobel_zhu::launch_schobel_zhu_up_and_out_option_diagonal_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::first_and_second>,
            model::equity::schobel_zhu::schobel_zhu_up_and_out_option_node_graph_workspace_bytes<OptionSide::call, pg::SensitivityOrders::first_and_second>,
            model::equity::schobel_zhu::launch_schobel_zhu_up_and_out_option_node_graph_sensitivities_cuda<OptionSide::call, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::schobel_zhu::prepare_up_and_out_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
