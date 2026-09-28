// Generated bates down_and_in_option gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/bates/product/down_and_in_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/down_and_in_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/down_and_in_option/down_and_in_options_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/down_and_in_puts/bates_01__down_and_in_puts_01__01_cartesian_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/bates/price_sensitivities/down_and_in_puts/bates_01__down_and_in_puts_01__01_cartesian_price_sensitivities_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/down_and_in_puts/bates_01__down_and_in_puts_01__01_cartesian_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/bates/prices/down_and_in_puts/bates_01__down_and_in_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_down_and_in_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::bates::prepare_bates_down_and_in_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "down_and_in_option", ""},
            11668826686056562688ULL, models, products, prepare,
            model::equity::bates::launch_bates_down_and_in_option_diagonal_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>,
            model::equity::bates::bates_down_and_in_option_node_graph_workspace_bytes<OptionSide::put, pg::SensitivityOrders::first_and_second>,
            model::equity::bates::launch_bates_down_and_in_option_node_graph_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::bates::prepare_down_and_in_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
