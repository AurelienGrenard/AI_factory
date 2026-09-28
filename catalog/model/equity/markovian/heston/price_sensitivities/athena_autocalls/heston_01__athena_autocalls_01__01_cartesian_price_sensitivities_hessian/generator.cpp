// Generated heston athena_autocall selected full-Hessian recipe.
#include "model/equity/markovian/heston/product/athena_autocall_price_gradients.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston/parameters/heston_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/markovian/heston/price_sensitivities/athena_autocalls/heston_01__athena_autocalls_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/heston/price_sensitivities/athena_autocalls/heston_01__athena_autocalls_01__01_cartesian_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/heston/price_sensitivities/athena_autocalls/heston_01__athena_autocalls_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/heston/prices/athena_autocalls/heston_01__athena_autocalls_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::heston::load_models(recipe.model_input);
        const auto products = product::load_athena_autocalls(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::heston::
                prepare_heston_athena_autocall_sensitivities(
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
                    "heston",
                    "athena_autocall",
                    "",
                },
                11668827016769044480ULL,
                models,
                products,
                prepare,
                model::equity::heston::
                    heston_athena_autocall_mixed_node_graph_workspace_bytes,
                model::equity::heston::
                    launch_heston_athena_autocall_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::heston::
                    prepare_athena_autocall_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
