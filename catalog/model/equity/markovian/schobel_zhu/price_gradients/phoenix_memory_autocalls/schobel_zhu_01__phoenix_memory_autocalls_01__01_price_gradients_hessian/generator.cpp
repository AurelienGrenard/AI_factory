// Generated schobel_zhu phoenix_memory_autocall selected full-Hessian recipe.
#include "model/equity/markovian/schobel_zhu/product/phoenix_memory_autocall_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json", "datasets/model/equity/markovian/schobel_zhu/price_gradients/phoenix_memory_autocalls/schobel_zhu_01__phoenix_memory_autocalls_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/schobel_zhu/price_gradients/phoenix_memory_autocalls/schobel_zhu_01__phoenix_memory_autocalls_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/price_gradients/phoenix_memory_autocalls/schobel_zhu_01__phoenix_memory_autocalls_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/schobel_zhu/prices/phoenix_memory_autocalls/schobel_zhu_01__phoenix_memory_autocalls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"product.autocall_barrier", {0.005, pg::BumpScale::relative}},
        {"product.coupon_barrier", {0.005, pg::BumpScale::relative}},
        {"product.protection_barrier", {0.005, pg::BumpScale::relative}},
        {"product.annual_coupon_rate", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_phoenix_memory_autocalls(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::schobel_zhu::
                prepare_schobel_zhu_phoenix_memory_autocall_sensitivities(
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
                    "schobel_zhu",
                    "phoenix_memory_autocall",
                    "",
                },
                11668827927302111232ULL,
                models,
                products,
                prepare,
                model::equity::schobel_zhu::
                    schobel_zhu_phoenix_memory_autocall_mixed_node_graph_workspace_bytes,
                model::equity::schobel_zhu::
                    launch_schobel_zhu_phoenix_memory_autocall_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::schobel_zhu::
                    prepare_phoenix_memory_autocall_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
