// Generated ${model} American gradient and diagonal Hessian with frozen exercise.
#include "model/equity/markovian/${model}/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/${model}/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog", "$url", "$source_recipe",
            PriceConstruction::$construction, {{
        $selections
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::$model::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::$model::prepare_${model}_american_option_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "$model", "american_option", ""},
            ${seed}ULL, models, products, prepare,
            model::equity::$model::launch_${model}_american_option_diagonal_sensitivities_cuda<
                OptionSide::$side, pg::SensitivityOrders::first_and_second>,
            model::equity::$model::${model}_american_option_node_graph_workspace_bytes<
                OptionSide::$side, pg::SensitivityOrders::first_and_second>,
            model::equity::$model::launch_${model}_american_option_node_graph_sensitivities_cuda<
                OptionSide::$side, pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::$model::prepare_american_option_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
