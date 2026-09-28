// Generated normal_inverse_gaussian forward_start_option selected full-Hessian recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/forward_start_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/forward_start_calls/normal_inverse_gaussian_01__forward_start_calls_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/forward_start_calls/normal_inverse_gaussian_01__forward_start_calls_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/price_sensitivities/forward_start_calls/normal_inverse_gaussian_01__forward_start_calls_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/forward_start_calls/normal_inverse_gaussian_01__forward_start_calls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.alpha", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"model.delta", {0.005, pg::BumpScale::relative}},
        {"product.moneyness", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::normal_inverse_gaussian::load_models(recipe.model_input);
        const auto products = product::load_forward_start_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::normal_inverse_gaussian::
                prepare_normal_inverse_gaussian_forward_start_option_sensitivities(
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
                    "normal_inverse_gaussian",
                    "forward_start_option",
                    "",
                },
                11668827618064465920ULL,
                models,
                products,
                prepare,
                model::equity::normal_inverse_gaussian::
                    normal_inverse_gaussian_forward_start_option_mixed_node_graph_workspace_bytes<OptionSide::call>,
                model::equity::normal_inverse_gaussian::
                    launch_normal_inverse_gaussian_forward_start_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::normal_inverse_gaussian::
                    prepare_forward_start_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
