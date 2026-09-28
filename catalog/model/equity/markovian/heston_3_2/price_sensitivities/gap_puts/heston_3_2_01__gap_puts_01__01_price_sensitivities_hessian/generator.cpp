// Generated heston_3_2 gap_option selected full-Hessian recipe.
#include "model/equity/markovian/heston_3_2/product/gap_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json", "datasets/product/gap_option/gap_put_options_01.json", "datasets/model/equity/markovian/heston_3_2/price_sensitivities/gap_puts/heston_3_2_01__gap_puts_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/heston_3_2/price_sensitivities/gap_puts/heston_3_2_01__gap_puts_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/heston_3_2/price_sensitivities/gap_puts/heston_3_2_01__gap_puts_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/heston_3_2/prices/gap_puts/heston_3_2_01__gap_puts_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.trigger_strike", {0.005, pg::BumpScale::relative}},
        {"product.payoff_strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::heston_3_2::load_models(recipe.model_input);
        const auto products = product::load_gap_options(recipe.product_input, OptionSide::put);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::heston_3_2::
                prepare_heston_3_2_gap_option_sensitivities(
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
                    "gap_option",
                    "",
                },
                11668827205747605504ULL,
                models,
                products,
                prepare,
                model::equity::heston_3_2::
                    heston_3_2_gap_option_mixed_node_graph_workspace_bytes<OptionSide::put>,
                model::equity::heston_3_2::
                    launch_heston_3_2_gap_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::heston_3_2::
                    prepare_gap_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
