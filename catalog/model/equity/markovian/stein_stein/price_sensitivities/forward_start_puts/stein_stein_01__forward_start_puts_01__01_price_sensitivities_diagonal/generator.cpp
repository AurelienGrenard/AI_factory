// Generated stein_stein forward_start_option gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/stein_stein/product/forward_start_option_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/markovian/stein_stein/price_sensitivities/forward_start_puts/stein_stein_01__forward_start_puts_01__01_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/stein_stein/price_sensitivities/forward_start_puts/stein_stein_01__forward_start_puts_01__01_price_sensitivities_diagonal/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/stein_stein/price_sensitivities/forward_start_puts/stein_stein_01__forward_start_puts_01__01_price_sensitivities_diagonal.json", "catalog/model/equity/markovian/stein_stein/prices/forward_start_puts/stein_stein_01__forward_start_puts_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.moneyness", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::stein_stein::load_models(recipe.model_input);
        const auto products = product::load_forward_start_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::stein_stein::prepare_stein_stein_forward_start_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "stein_stein", "forward_start_option", ""},
            11668828030381326336ULL, models, products, prepare,
            model::equity::stein_stein::launch_stein_stein_forward_start_option_diagonal_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>,
            model::equity::stein_stein::stein_stein_forward_start_option_node_graph_workspace_bytes<OptionSide::put, pg::SensitivityOrders::first_and_second>,
            model::equity::stein_stein::launch_stein_stein_forward_start_option_node_graph_sensitivities_cuda<OptionSide::put, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::stein_stein::prepare_forward_start_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
