// Generated sabr lookback_option selected full-Hessian recipe.
#include "model/equity/markovian/sabr/product/lookback_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/lookback_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/lookback_option/lookback_options_01.json", "datasets/model/equity/markovian/sabr/price_gradients/lookback_options/sabr_01__lookback_options_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/sabr/price_gradients/lookback_options/sabr_01__lookback_options_01__01_cartesian_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_gradients/lookback_options/sabr_01__lookback_options_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/sabr/prices/lookback_options/sabr_01__lookback_options_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::sabr::load_models(recipe.model_input);
        const auto products = product::load_lookback_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::sabr::
                prepare_sabr_lookback_option_sensitivities(
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
                    "sabr",
                    "lookback_option",
                    "",
                },
                11668827776978255872ULL,
                models,
                products,
                prepare,
                model::equity::sabr::
                    sabr_lookback_option_mixed_node_graph_workspace_bytes,
                model::equity::sabr::
                    launch_sabr_lookback_option_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::sabr::
                    prepare_lookback_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
