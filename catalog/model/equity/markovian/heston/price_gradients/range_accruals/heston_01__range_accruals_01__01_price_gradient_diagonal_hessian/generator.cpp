// Generated heston range_accrual gradient and diagonal-Hessian recipe.
#include "model/equity/markovian/heston/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston/parameters/heston_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/markovian/heston/price_gradients/range_accruals/heston_01__range_accruals_01__01_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/heston/price_gradients/range_accruals/heston_01__range_accruals_01__01_price_gradient_diagonal_hessian/generation.yaml", "https://datasets.ai-factory.example/v2/model/equity/markovian/heston/price_gradients/range_accruals/heston_01__range_accruals_01__01_price_gradient_diagonal_hessian.json", "catalog/model/equity/markovian/heston/prices/range_accruals/heston_01__range_accruals_01__01/recipe.yaml",
            PriceConstruction::Aligned, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}},
        {"product.coupon_rate", {0.0005, pg::BumpScale::absolute}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second};
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models = model::equity::heston::load_models(recipe.model_input);
        const auto products = product::load_range_accruals(recipe.product_input);
        const auto prepare = [](const auto& model_rows, const auto& product_rows,
                                PriceConstruction construction, pg::TimeConfiguration time,
                                const pg::PriceGradientConfiguration& selected) {
            return model::equity::heston::prepare_heston_range_accrual_sensitivities(
                model_rows, product_rows, construction, time, selected,
                {pg::SensitivityOrders::first_and_second});
        };
        return offline::pricing::price_gradients::execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second>(
            recipe,
            {offline::cuda_tuning::PricingFamily::equity_step_mc, "heston", "range_accrual", ""},
            11668827098373423104ULL, models, products, prepare,
            model::equity::heston::launch_heston_range_accrual_diagonal_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            model::equity::heston::heston_range_accrual_node_graph_workspace_bytes<pg::SensitivityOrders::first_and_second>,
            model::equity::heston::launch_heston_range_accrual_node_graph_sensitivities_cuda<pg::SensitivityOrders::first_and_second>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::heston::prepare_range_accrual_diagonal_sensitivity_stencils_cuda);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
