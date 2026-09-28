// Generated stein_stein gap_option selected full-Hessian recipe.
#include "model/equity/markovian/stein_stein/product/gap_option_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json", "datasets/product/gap_option/gap_call_options_01.json", "datasets/model/equity/markovian/stein_stein/price_sensitivities/gap_calls/stein_stein_01__gap_calls_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/stein_stein/price_sensitivities/gap_calls/stein_stein_01__gap_calls_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/stein_stein/price_sensitivities/gap_calls/stein_stein_01__gap_calls_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/stein_stein/prices/gap_calls/stein_stein_01__gap_calls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.trigger_strike", {0.005, pg::BumpScale::relative}},
        {"product.payoff_strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::stein_stein::load_models(recipe.model_input);
        const auto products = product::load_gap_options(recipe.product_input, OptionSide::call);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::stein_stein::
                prepare_stein_stein_gap_option_sensitivities(
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
                    "stein_stein",
                    "gap_option",
                    "",
                },
                11668828034676293632ULL,
                models,
                products,
                prepare,
                model::equity::stein_stein::
                    stein_stein_gap_option_mixed_node_graph_workspace_bytes<OptionSide::call>,
                model::equity::stein_stein::
                    launch_stein_stein_gap_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::stein_stein::
                    prepare_gap_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
