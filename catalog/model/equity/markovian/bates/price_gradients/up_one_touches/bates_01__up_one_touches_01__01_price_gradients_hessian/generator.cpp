// Generated bates up_one_touch selected full-Hessian recipe.
#include "model/equity/markovian/bates/product/up_one_touch_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/markovian/bates/price_gradients/up_one_touches/bates_01__up_one_touches_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/bates/price_gradients/up_one_touches/bates_01__up_one_touches_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_gradients/up_one_touches/bates_01__up_one_touches_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/bates/prices/up_one_touches/bates_01__up_one_touches_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_up_one_touches(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::bates::
                prepare_bates_up_one_touch_sensitivities(
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
                    "bates",
                    "up_one_touch",
                    "",
                },
                11668826763365974016ULL,
                models,
                products,
                prepare,
                model::equity::bates::
                    bates_up_one_touch_mixed_node_graph_workspace_bytes,
                model::equity::bates::
                    launch_bates_up_one_touch_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::bates::
                    prepare_up_one_touch_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
