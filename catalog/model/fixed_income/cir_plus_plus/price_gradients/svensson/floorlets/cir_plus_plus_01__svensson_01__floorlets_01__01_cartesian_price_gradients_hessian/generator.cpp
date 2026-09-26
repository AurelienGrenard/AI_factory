// Generated cir_plus_plus rate_option selected full-Hessian recipe.
#include "model/fixed_income/cir_plus_plus/product/svensson/rate_option_price_gradients.cuh"
#include "model/fixed_income/cir_plus_plus/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/rate_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::cir_plus_plus::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json", "datasets/product/rate_option/rate_options_01.json", "datasets/model/fixed_income/cir_plus_plus/price_gradients/svensson/floorlets/cir_plus_plus_01__svensson_01__floorlets_01__01_cartesian_price_gradients_hessian.json", "catalog/model/fixed_income/cir_plus_plus/price_gradients/svensson/floorlets/cir_plus_plus_01__svensson_01__floorlets_01__01_cartesian_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/cir_plus_plus/price_gradients/svensson/floorlets/cir_plus_plus_01__svensson_01__floorlets_01__01_cartesian_price_gradients_hessian.json", "catalog/model/fixed_income/cir_plus_plus/prices/svensson/floorlets/cir_plus_plus_01__svensson_01__floorlets_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_term_mean", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta0", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta1", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta2", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta3", {0.0005, pg::BumpScale::absolute}},
        {"curve.tau1", {0.005, pg::BumpScale::relative}},
        {"curve.tau2", {0.005, pg::BumpScale::relative}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        recipe.curve_input = "datasets/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::cir_plus_plus::load_models(recipe.model_input);
        const auto curves = curve::svensson::load_curves(recipe.curve_input);
        const auto products =
            product::load_rate_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_cir_plus_plus_svensson_rate_option_sensitivities(
                    model_rows,
                    curve_rows, product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        return offline::pricing::price_gradients::
            execute_curve_mixed_node_graph_dataset<false>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::closed_form,
                    "cir_plus_plus",
                    "rate_option",
                    "svensson",
                },
                0ULL,
                models,
                curves, products,
                prepare,
                model_namespace::
                    cir_plus_plus_svensson_rate_option_mixed_node_graph_workspace_bytes<
                        OptionSide::put>,
                model_namespace::
                    launch_cir_plus_plus_svensson_rate_option_mixed_node_graph_sensitivities_cuda<
                        OptionSide::put>,
                0U,
                model_namespace::
                    prepare_rate_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
