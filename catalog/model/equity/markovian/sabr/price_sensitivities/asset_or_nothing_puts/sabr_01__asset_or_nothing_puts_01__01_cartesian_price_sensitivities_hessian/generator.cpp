// Generated sabr asset_or_nothing_option selected full-Hessian recipe.
#include "model/equity/markovian/sabr/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/markovian/sabr/price_sensitivities/asset_or_nothing_puts/sabr_01__asset_or_nothing_puts_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/sabr/price_sensitivities/asset_or_nothing_puts/sabr_01__asset_or_nothing_puts_01__01_cartesian_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/price_sensitivities/asset_or_nothing_puts/sabr_01__asset_or_nothing_puts_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/sabr/prices/asset_or_nothing_puts/sabr_01__asset_or_nothing_puts_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
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
        const auto products = product::load_asset_or_nothing_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::sabr::
                prepare_sabr_asset_or_nothing_option_sensitivities(
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
                    "asset_or_nothing_option",
                    "",
                },
                11668827703963811840ULL,
                models,
                products,
                prepare,
                model::equity::sabr::
                    sabr_asset_or_nothing_option_mixed_node_graph_workspace_bytes<OptionSide::put>,
                model::equity::sabr::
                    launch_sabr_asset_or_nothing_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::sabr::
                    prepare_asset_or_nothing_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
