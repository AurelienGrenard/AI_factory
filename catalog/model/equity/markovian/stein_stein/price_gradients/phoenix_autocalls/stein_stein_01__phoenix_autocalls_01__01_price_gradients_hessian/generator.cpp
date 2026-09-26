// Generated stein_stein phoenix_autocall selected full-Hessian recipe.
#include "model/equity/markovian/stein_stein/product/phoenix_autocall_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/stein_stein/price_gradients/phoenix_autocalls/stein_stein_01__phoenix_autocalls_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/stein_stein/price_gradients/phoenix_autocalls/stein_stein_01__phoenix_autocalls_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/stein_stein/price_gradients/phoenix_autocalls/stein_stein_01__phoenix_autocalls_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/stein_stein/prices/phoenix_autocalls/stein_stein_01__phoenix_autocalls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.coupon_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::stein_stein::load_models(recipe.model_input);
        const auto products = product::load_phoenix_autocalls(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::stein_stein::
                prepare_stein_stein_phoenix_autocall_sensitivities(
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
                    "phoenix_autocall",
                    "",
                },
                11668828056151130112ULL,
                models,
                products,
                prepare,
                model::equity::stein_stein::
                    stein_stein_phoenix_autocall_mixed_node_graph_workspace_bytes,
                model::equity::stein_stein::
                    launch_stein_stein_phoenix_autocall_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::stein_stein::
                    prepare_phoenix_autocall_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
