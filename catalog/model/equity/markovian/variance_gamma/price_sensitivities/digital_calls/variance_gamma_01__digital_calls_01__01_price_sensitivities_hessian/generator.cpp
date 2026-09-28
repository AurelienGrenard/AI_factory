// Generated variance_gamma digital_option selected full-Hessian recipe.
#include "model/equity/markovian/variance_gamma/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/markovian/variance_gamma/price_sensitivities/digital_calls/variance_gamma_01__digital_calls_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/variance_gamma/price_sensitivities/digital_calls/variance_gamma_01__digital_calls_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/price_sensitivities/digital_calls/variance_gamma_01__digital_calls_01__01_price_sensitivities_hessian.json", "catalog/model/equity/markovian/variance_gamma/prices/digital_calls/variance_gamma_01__digital_calls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.nu", {0.005, pg::BumpScale::relative}},
        {"model.theta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::variance_gamma::load_models(recipe.model_input);
        const auto products = product::load_digital_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::variance_gamma::
                prepare_variance_gamma_digital_option_sensitivities(
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
                    "digital_option",
                    "",
                },
                11668828133460541440ULL,
                models,
                products,
                prepare,
                model::equity::variance_gamma::
                    variance_gamma_digital_option_mixed_node_graph_workspace_bytes<OptionSide::call>,
                model::equity::variance_gamma::
                    launch_variance_gamma_digital_option_mixed_node_graph_sensitivities_cuda<OptionSide::call>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::variance_gamma::
                    prepare_digital_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
