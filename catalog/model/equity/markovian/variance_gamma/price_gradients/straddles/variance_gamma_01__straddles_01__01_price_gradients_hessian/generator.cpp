// Generated variance_gamma straddle selected full-Hessian recipe.
#include "model/equity/markovian/variance_gamma/product/straddle_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/markovian/variance_gamma/price_gradients/straddles/variance_gamma_01__straddles_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/variance_gamma/price_gradients/straddles/variance_gamma_01__straddles_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_gradients/straddles/variance_gamma_01__straddles_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/variance_gamma/prices/straddles/variance_gamma_01__straddles_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_straddles(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::variance_gamma::
                prepare_variance_gamma_straddle_sensitivities(
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
                    "variance_gamma",
                    "straddle",
                    "",
                },
                11668828210769952768ULL,
                models,
                products,
                prepare,
                model::equity::variance_gamma::
                    variance_gamma_straddle_mixed_node_graph_workspace_bytes,
                model::equity::variance_gamma::
                    launch_variance_gamma_straddle_mixed_node_graph_sensitivities_cuda,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::variance_gamma::
                    prepare_straddle_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
