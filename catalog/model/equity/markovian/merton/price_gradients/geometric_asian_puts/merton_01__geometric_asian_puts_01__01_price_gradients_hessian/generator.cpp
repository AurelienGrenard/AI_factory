// Generated merton geometric_asian_option selected full-Hessian recipe.
#include "model/equity/markovian/merton/product/geometric_asian_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/markovian/merton/price_gradients/geometric_asian_puts/merton_01__geometric_asian_puts_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/merton/price_gradients/geometric_asian_puts/merton_01__geometric_asian_puts_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_gradients/geometric_asian_puts/merton_01__geometric_asian_puts_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/merton/prices/geometric_asian_puts/merton_01__geometric_asian_puts_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_geometric_asian_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::merton::
                prepare_merton_geometric_asian_option_sensitivities(
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
                    "geometric_asian_option",
                    "",
                },
                11668827497805381632ULL,
                models,
                products,
                prepare,
                model::equity::merton::
                    merton_geometric_asian_option_mixed_node_graph_workspace_bytes<OptionSide::put>,
                model::equity::merton::
                    launch_merton_geometric_asian_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::merton::
                    prepare_geometric_asian_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
