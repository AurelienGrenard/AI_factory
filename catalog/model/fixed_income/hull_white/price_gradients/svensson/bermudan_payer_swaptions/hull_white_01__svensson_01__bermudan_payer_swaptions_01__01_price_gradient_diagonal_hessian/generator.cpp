// Generated hull_white Bermudan-swaption gradient and diagonal-Hessian recipe.
#include "model/fixed_income/hull_white/product/svensson/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/hull_white/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::hull_white::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/hull_white/parameters/hull_white_01.json", "datasets/product/bermudan_swaption/bermudan_swaptions_01.json", "datasets/model/fixed_income/hull_white/price_gradients/svensson/bermudan_payer_swaptions/hull_white_01__svensson_01__bermudan_payer_swaptions_01__01_price_gradient_diagonal_hessian.json", "catalog/model/fixed_income/hull_white/price_gradients/svensson/bermudan_payer_swaptions/hull_white_01__svensson_01__bermudan_payer_swaptions_01__01_price_gradient_diagonal_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/fixed_income/hull_white/price_gradients/svensson/bermudan_payer_swaptions/hull_white_01__svensson_01__bermudan_payer_swaptions_01__01_price_gradient_diagonal_hessian.json", "catalog/model/fixed_income/hull_white/prices/svensson/bermudan_payer_swaptions/hull_white_01__svensson_01__bermudan_payer_swaptions_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"curve.beta0", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta1", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta2", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta3", {0.0005, pg::BumpScale::absolute}},
        {"curve.tau1", {0.005, pg::BumpScale::relative}},
        {"curve.tau2", {0.005, pg::BumpScale::relative}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        recipe.curve_input = "datasets/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::hull_white::load_models(recipe.model_input);
        const auto curves = curve::svensson::load_curves(recipe.curve_input);
        const auto products =
            product::load_bermudan_swaptions(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_hull_white_svensson_bermudan_swaption_sensitivities(
                    model_rows, curve_rows, product_rows,
                    construction, time, configuration,
                    {pg::SensitivityOrders::first_and_second}
                );
        };
        return offline::pricing::price_gradients::
            execute_curve_node_graph_dataset<
            pg::SensitivityOrders::first_and_second
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::gaussian_rate_lsm,
                "hull_white",
                "bermudan_swaption",
                "svensson",
            },
            11668829108418117632ULL,
            models,
            curves, products,
            prepare,
            model_namespace::
                launch_hull_white_svensson_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >,
            model_namespace::
                hull_white_svensson_bermudan_swaption_node_graph_workspace_bytes<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >,
            model_namespace::
                launch_hull_white_svensson_bermudan_swaption_node_graph_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model_namespace::
                prepare_bermudan_swaption_diagonal_sensitivity_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
