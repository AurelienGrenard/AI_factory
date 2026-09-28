// Generated heston_3_2 cliquet selected full-Hessian recipe.
#include "model/equity/markovian/heston_3_2/product/cliquet_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/markovian/heston_3_2/price_sensitivities/cliquets/heston_3_2_01__cliquets_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/heston_3_2/price_sensitivities/cliquets/heston_3_2_01__cliquets_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/heston_3_2/price_sensitivities/cliquets/heston_3_2_01__cliquets_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/heston_3_2/prices/cliquets/heston_3_2_01__cliquets_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.participation_rate", {0.005, pg::BumpScale::relative}},
        {"product.local_floor", {0.002, pg::BumpScale::absolute}},
        {"product.local_cap", {0.002, pg::BumpScale::absolute}},
        {"product.global_floor", {0.002, pg::BumpScale::absolute}},
        {"product.global_cap", {0.002, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::heston_3_2::load_models(recipe.model_input);
        const auto products = product::load_cliquets(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::heston_3_2::
                prepare_heston_3_2_cliquet_sensitivities(
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
                    "heston_3_2",
                    "cliquet",
                    "",
                },
                11668827154207997952ULL,
                models,
                products,
                prepare,
                model::equity::heston_3_2::
                    heston_3_2_cliquet_mixed_node_graph_workspace_bytes,
                model::equity::heston_3_2::
                    launch_heston_3_2_cliquet_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::heston_3_2::
                    prepare_cliquet_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
