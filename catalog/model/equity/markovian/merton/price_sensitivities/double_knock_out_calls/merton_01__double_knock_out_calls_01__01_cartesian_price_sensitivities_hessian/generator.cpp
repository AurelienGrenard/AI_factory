// Generated merton double_knock_out_option selected full-Hessian recipe.
#include "model/equity/markovian/merton/product/double_knock_out_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/double_knock_out_option/double_knock_out_options_01.json", "datasets/model/equity/markovian/merton/price_sensitivities/double_knock_out_calls/merton_01__double_knock_out_calls_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/merton/price_sensitivities/double_knock_out_calls/merton_01__double_knock_out_calls_01__01_cartesian_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_sensitivities/double_knock_out_calls/merton_01__double_knock_out_calls_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/merton/prices/double_knock_out_calls/merton_01__double_knock_out_calls_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_double_knock_out_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::merton::
                prepare_merton_double_knock_out_option_sensitivities(
                    model_rows,
                    product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        return offline::pricing::price_gradients::
            execute_mixed_node_graph_dataset<true>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::equity_step_mc,
                    "merton",
                    "double_knock_out_option",
                    "",
                },
                11668827450560741376ULL,
                models,
                products,
                prepare,
                model::equity::merton::
                    merton_double_knock_out_option_mixed_node_graph_workspace_bytes<OptionSide::call>,
                model::equity::merton::
                    launch_merton_double_knock_out_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::merton::
                    prepare_double_knock_out_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
