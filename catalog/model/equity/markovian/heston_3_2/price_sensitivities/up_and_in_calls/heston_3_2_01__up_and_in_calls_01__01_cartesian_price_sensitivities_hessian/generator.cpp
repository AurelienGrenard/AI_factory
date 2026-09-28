// Generated heston_3_2 up_and_in_option selected full-Hessian recipe.
#include "model/equity/markovian/heston_3_2/product/up_and_in_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "product/up_and_in_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json", "datasets/product/up_and_in_option/up_and_in_options_01.json", "datasets/model/equity/markovian/heston_3_2/price_sensitivities/up_and_in_calls/heston_3_2_01__up_and_in_calls_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/heston_3_2/price_sensitivities/up_and_in_calls/heston_3_2_01__up_and_in_calls_01__01_cartesian_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/heston_3_2/price_sensitivities/up_and_in_calls/heston_3_2_01__up_and_in_calls_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/equity/markovian/heston_3_2/prices/up_and_in_calls/heston_3_2_01__up_and_in_calls_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::heston_3_2::load_models(recipe.model_input);
        const auto products = product::load_up_and_in_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::heston_3_2::
                prepare_heston_3_2_up_and_in_option_sensitivities(
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
                    "up_and_in_option",
                    "",
                },
                11668827240107343872ULL,
                models,
                products,
                prepare,
                model::equity::heston_3_2::
                    heston_3_2_up_and_in_option_mixed_node_graph_workspace_bytes<OptionSide::call>,
                model::equity::heston_3_2::
                    launch_heston_3_2_up_and_in_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::heston_3_2::
                    prepare_up_and_in_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
