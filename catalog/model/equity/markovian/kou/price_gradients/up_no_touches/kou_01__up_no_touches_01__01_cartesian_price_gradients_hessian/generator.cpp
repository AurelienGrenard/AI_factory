// Generated kou up_no_touch selected full-Hessian recipe.
#include "model/equity/markovian/kou/product/up_no_touch_price_gradients.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "product/up_no_touch/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/kou/parameters/kou_01.json", "datasets/product/up_no_touch/up_no_touches_01.json", "datasets/model/equity/markovian/kou/price_gradients/up_no_touches/kou_01__up_no_touches_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/kou/price_gradients/up_no_touches/kou_01__up_no_touches_01__01_cartesian_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/kou/price_gradients/up_no_touches/kou_01__up_no_touches_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/kou/prices/up_no_touches/kou_01__up_no_touches_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.up_probability", {0.002, pg::BumpScale::absolute}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::kou::load_models(recipe.model_input);
        const auto products = product::load_up_no_touches(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::kou::
                prepare_kou_up_no_touch_sensitivities(
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
                    "kou",
                    "up_no_touch",
                    "",
                },
                11668827390431199232ULL,
                models,
                products,
                prepare,
                model::equity::kou::
                    kou_up_no_touch_mixed_node_graph_workspace_bytes,
                model::equity::kou::
                    launch_kou_up_no_touch_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::kou::
                    prepare_up_no_touch_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
