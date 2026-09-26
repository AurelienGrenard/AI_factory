// Generated merton forward_start_option selected full-Hessian recipe.
#include "model/equity/markovian/merton/product/forward_start_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/markovian/merton/price_gradients/forward_start_puts/merton_01__forward_start_puts_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/merton/price_gradients/forward_start_puts/merton_01__forward_start_puts_01__01_cartesian_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_gradients/forward_start_puts/merton_01__forward_start_puts_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/merton/prices/forward_start_puts/merton_01__forward_start_puts_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.moneyness", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_forward_start_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::merton::
                prepare_merton_forward_start_option_sensitivities(
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
                    "merton",
                    "forward_start_option",
                    "",
                },
                11668827480625512448ULL,
                models,
                products,
                prepare,
                model::equity::merton::
                    merton_forward_start_option_mixed_node_graph_workspace_bytes<OptionSide::put>,
                model::equity::merton::
                    launch_merton_forward_start_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::merton::
                    prepare_forward_start_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
