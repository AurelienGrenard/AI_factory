// Generated black_scholes up_no_touch selected full-Hessian recipe.
#include "model/equity/markovian/black_scholes/product/up_no_touch_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/up_no_touch/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/up_no_touch/up_no_touches_01.json", "datasets/model/equity/markovian/black_scholes/price_sensitivities/up_no_touches/black_scholes_01__up_no_touches_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/black_scholes/price_sensitivities/up_no_touches/black_scholes_01__up_no_touches_01__01_cartesian_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/black_scholes/price_sensitivities/up_no_touches/black_scholes_01__up_no_touches_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/black_scholes/prices/up_no_touches/black_scholes_01__up_no_touches_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_up_no_touches(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::black_scholes::
                prepare_black_scholes_up_no_touch_sensitivities(
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
                    "black_scholes",
                    "up_no_touch",
                    "",
                },
                11668826832085450752ULL,
                models,
                products,
                prepare,
                model::equity::black_scholes::
                    black_scholes_up_no_touch_mixed_node_graph_workspace_bytes,
                model::equity::black_scholes::
                    launch_black_scholes_up_no_touch_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::black_scholes::
                    prepare_up_no_touch_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
