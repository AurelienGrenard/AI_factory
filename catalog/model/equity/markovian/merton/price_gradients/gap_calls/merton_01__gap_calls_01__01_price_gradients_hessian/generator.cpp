// Generated merton gap_option selected full-Hessian recipe.
#include "model/equity/markovian/merton/product/gap_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/gap_option/gap_call_options_01.json", "datasets/model/equity/markovian/merton/price_gradients/gap_calls/merton_01__gap_calls_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/merton/price_gradients/gap_calls/merton_01__gap_calls_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/price_gradients/gap_calls/merton_01__gap_calls_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/merton/prices/gap_calls/merton_01__gap_calls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.trigger_strike", {0.005, pg::BumpScale::relative}},
        {"product.payoff_strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::merton::load_models(recipe.model_input);
        const auto products = product::load_gap_options(recipe.product_input, OptionSide::call);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::merton::
                prepare_merton_gap_option_sensitivities(
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
                    "gap_option",
                    "",
                },
                11668827484920479744ULL,
                models,
                products,
                prepare,
                model::equity::merton::
                    merton_gap_option_mixed_node_graph_workspace_bytes<OptionSide::call>,
                model::equity::merton::
                    launch_merton_gap_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::merton::
                    prepare_gap_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
