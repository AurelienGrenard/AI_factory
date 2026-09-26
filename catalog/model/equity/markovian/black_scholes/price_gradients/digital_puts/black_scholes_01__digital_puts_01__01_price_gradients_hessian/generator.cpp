// Generated black_scholes digital_option selected full-Hessian recipe.
#include "model/equity/markovian/black_scholes/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/markovian/black_scholes/price_gradients/digital_puts/black_scholes_01__digital_puts_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/black_scholes/price_gradients/digital_puts/black_scholes_01__digital_puts_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/equity/markovian/black_scholes/price_gradients/digital_puts/black_scholes_01__digital_puts_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/black_scholes/prices/digital_puts/black_scholes_01__digital_puts_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_digital_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::black_scholes::
                prepare_black_scholes_digital_option_sensitivities(
                    model_rows,
                    product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        return offline::pricing::price_gradients::
            execute_mixed_node_graph_dataset<false>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::closed_form,
                    "black_scholes",
                    "digital_option",
                    "",
                },
                0ULL,
                models,
                products,
                prepare,
                model::equity::black_scholes::
                    black_scholes_digital_option_mixed_node_graph_workspace_bytes<OptionSide::put>,
                model::equity::black_scholes::
                    launch_black_scholes_digital_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>,
                0U,
                model::equity::black_scholes::
                    prepare_digital_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
