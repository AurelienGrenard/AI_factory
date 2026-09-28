// Generated variance_gamma up_one_touch selected full-Hessian recipe.
#include "model/equity/markovian/variance_gamma/product/up_one_touch_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/markovian/variance_gamma/price_sensitivities/up_one_touches/variance_gamma_01__up_one_touches_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/variance_gamma/price_sensitivities/up_one_touches/variance_gamma_01__up_one_touches_01__01_cartesian_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_sensitivities/up_one_touches/variance_gamma_01__up_one_touches_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/variance_gamma/prices/up_one_touches/variance_gamma_01__up_one_touches_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_up_one_touches(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::variance_gamma::
                prepare_variance_gamma_up_one_touch_sensitivities(
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
                    "variance_gamma",
                    "up_one_touch",
                    "",
                },
                11668828227949821952ULL,
                models,
                products,
                prepare,
                model::equity::variance_gamma::
                    variance_gamma_up_one_touch_mixed_node_graph_workspace_bytes,
                model::equity::variance_gamma::
                    launch_variance_gamma_up_one_touch_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::variance_gamma::
                    prepare_up_one_touch_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
