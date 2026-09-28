// Generated bates forward_start_option selected full-Hessian recipe.
#include "model/equity/markovian/bates/product/forward_start_option_price_gradients.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/markovian/bates/price_sensitivities/forward_start_calls/bates_01__forward_start_calls_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/bates/price_sensitivities/forward_start_calls/bates_01__forward_start_calls_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/price_sensitivities/forward_start_calls/bates_01__forward_start_calls_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/bates/prices/forward_start_calls/bates_01__forward_start_calls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.jump_intensity", {0.05, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {0.002, pg::BumpScale::absolute}},
        {"product.moneyness", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::bates::load_models(recipe.model_input);
        const auto products = product::load_forward_start_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::bates::
                prepare_bates_forward_start_option_sensitivities(
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
                    "bates",
                    "forward_start_option",
                    "",
                },
                11668826703236431872ULL,
                models,
                products,
                prepare,
                model::equity::bates::
                    bates_forward_start_option_mixed_node_graph_workspace_bytes<OptionSide::call>,
                model::equity::bates::
                    launch_bates_forward_start_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::bates::
                    prepare_forward_start_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
