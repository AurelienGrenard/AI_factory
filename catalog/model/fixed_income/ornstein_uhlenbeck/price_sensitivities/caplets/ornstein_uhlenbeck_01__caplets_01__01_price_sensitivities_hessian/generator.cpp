// Generated ornstein_uhlenbeck rate_option selected full-Hessian recipe.
#include "model/fixed_income/ornstein_uhlenbeck/product/rate_option_price_gradients.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "product/rate_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::ornstein_uhlenbeck;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01.json", "datasets/product/rate_option/rate_options_01.json", "datasets/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/caplets/ornstein_uhlenbeck_01__caplets_01__01_price_sensitivities_hessian.json", "catalog/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/caplets/ornstein_uhlenbeck_01__caplets_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/caplets/ornstein_uhlenbeck_01__caplets_01__01_price_sensitivities_hessian.json", "catalog/model/fixed_income/ornstein_uhlenbeck/prices/caplets/ornstein_uhlenbeck_01__caplets_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::fixed_income::ornstein_uhlenbeck::load_models(recipe.model_input);
        const auto products =
            product::load_rate_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_ornstein_uhlenbeck_rate_option_sensitivities(
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
                    "ornstein_uhlenbeck",
                    "rate_option",
                    "",
                },
                0ULL,
                models,
                products,
                prepare,
                model_namespace::
                    ornstein_uhlenbeck_rate_option_mixed_node_graph_workspace_bytes<
                        OptionSide::call>,
                model_namespace::
                    launch_ornstein_uhlenbeck_rate_option_mixed_node_graph_sensitivities_cuda<
                        OptionSide::call>,
                0U,
                model_namespace::
                    prepare_rate_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
