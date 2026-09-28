// Generated merton up_no_touch selected full-Hessian recipe.
#include "model/equity/markovian/merton/product/up_no_touch_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/up_no_touch/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/up_no_touch/up_no_touches_01.json", "datasets/model/equity/markovian/merton/price_sensitivities/up_no_touches/merton_01__up_no_touches_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/merton/price_sensitivities/up_no_touches/merton_01__up_no_touches_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_sensitivities/up_no_touches/merton_01__up_no_touches_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/merton/prices/up_no_touches/merton_01__up_no_touches_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_up_no_touches(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::merton::
                prepare_merton_up_no_touch_sensitivities(
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
                    "up_no_touch",
                    "",
                },
                11668827532165120000ULL,
                models,
                products,
                prepare,
                model::equity::merton::
                    merton_up_no_touch_mixed_node_graph_workspace_bytes,
                model::equity::merton::
                    launch_merton_up_no_touch_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::merton::
                    prepare_up_no_touch_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
