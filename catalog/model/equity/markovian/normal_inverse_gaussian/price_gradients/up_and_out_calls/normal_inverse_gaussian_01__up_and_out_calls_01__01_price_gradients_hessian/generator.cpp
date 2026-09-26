// Generated normal_inverse_gaussian up_and_out_option selected full-Hessian recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/up_and_out_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/up_and_out_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/up_and_out_option/up_and_out_options_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_gradients/up_and_out_calls/normal_inverse_gaussian_01__up_and_out_calls_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_gradients/up_and_out_calls/normal_inverse_gaussian_01__up_and_out_calls_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_gradients/up_and_out_calls/normal_inverse_gaussian_01__up_and_out_calls_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/up_and_out_calls/normal_inverse_gaussian_01__up_and_out_calls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.alpha", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.delta", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_up_and_out_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::normal_inverse_gaussian::
                prepare_normal_inverse_gaussian_up_and_out_option_sensitivities(
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
                    "normal_inverse_gaussian",
                    "up_and_out_option",
                    "",
                },
                11668827669604073472ULL,
                models,
                products,
                prepare,
                model::equity::normal_inverse_gaussian::
                    normal_inverse_gaussian_up_and_out_option_mixed_node_graph_workspace_bytes<OptionSide::call>,
                model::equity::normal_inverse_gaussian::
                    launch_normal_inverse_gaussian_up_and_out_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::normal_inverse_gaussian::
                    prepare_up_and_out_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
