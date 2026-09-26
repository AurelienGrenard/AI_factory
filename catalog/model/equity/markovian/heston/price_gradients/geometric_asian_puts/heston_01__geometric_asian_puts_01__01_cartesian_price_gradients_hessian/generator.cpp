// Generated heston geometric_asian_option selected full-Hessian recipe.
#include "model/equity/markovian/heston/product/geometric_asian_option_price_gradients.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston/parameters/heston_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/markovian/heston/price_gradients/geometric_asian_puts/heston_01__geometric_asian_puts_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/heston/price_gradients/geometric_asian_puts/heston_01__geometric_asian_puts_01__01_cartesian_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/heston/price_gradients/geometric_asian_puts/heston_01__geometric_asian_puts_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/heston/prices/geometric_asian_puts/heston_01__geometric_asian_puts_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::heston::load_models(recipe.model_input);
        const auto products = product::load_geometric_asian_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::heston::
                prepare_heston_geometric_asian_option_sensitivities(
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
                    "heston",
                    "geometric_asian_option",
                    "",
                },
                11668827081193553920ULL,
                models,
                products,
                prepare,
                model::equity::heston::
                    heston_geometric_asian_option_mixed_node_graph_workspace_bytes<OptionSide::put>,
                model::equity::heston::
                    launch_heston_geometric_asian_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::heston::
                    prepare_geometric_asian_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
