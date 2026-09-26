// Generated normal_inverse_gaussian range_accrual selected full-Hessian recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_gradients/range_accruals/normal_inverse_gaussian_01__range_accruals_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_gradients/range_accruals/normal_inverse_gaussian_01__range_accruals_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_gradients/range_accruals/normal_inverse_gaussian_01__range_accruals_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/range_accruals/normal_inverse_gaussian_01__range_accruals_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.alpha", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.delta", {0.005, pg::BumpScale::relative}},
        {"product.lower_barrier", {0.005, pg::BumpScale::relative}},
        {"product.upper_barrier", {0.005, pg::BumpScale::relative}},
        {"product.coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_range_accruals(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::normal_inverse_gaussian::
                prepare_normal_inverse_gaussian_range_accrual_sensitivities(
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
                    offline::cuda_tuning::PricingFamily::equity_exact_mc,
                    "normal_inverse_gaussian",
                    "range_accrual",
                    "",
                },
                11668827656719171584ULL,
                models,
                products,
                prepare,
                model::equity::normal_inverse_gaussian::
                    normal_inverse_gaussian_range_accrual_mixed_node_graph_workspace_bytes,
                model::equity::normal_inverse_gaussian::
                    launch_normal_inverse_gaussian_range_accrual_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::normal_inverse_gaussian::
                    prepare_range_accrual_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
