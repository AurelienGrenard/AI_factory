// Generated g2_plus_plus zero-coupon bond barrier call first-order gradients.
#include "model/fixed_income/g2_plus_plus/product/svensson/zero_coupon_bond_up_and_out_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/zero_coupon_bond_up_and_out/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2_plus_plus::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json", "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01_shard00.json", "datasets/model/fixed_income/g2_plus_plus/price_gradients/svensson/zero_coupon_bond_up_and_out_calls/g2_plus_plus_01__svensson_01__zero_coupon_bond_up_and_out_calls_01__01_cartesian_shard00_price_gradient.json", "catalog/model/fixed_income/g2_plus_plus/price_gradients/svensson/zero_coupon_bond_up_and_out_calls/g2_plus_plus_01__svensson_01__zero_coupon_bond_up_and_out_calls_01__01_cartesian_shard00_price_gradient/generation.yaml",
            "catalog/model/fixed_income/g2_plus_plus/prices/svensson/zero_coupon_bond_up_and_out_calls/g2_plus_plus_01__svensson_01__zero_coupon_bond_up_and_out_calls_01__01_cartesian_shard00/recipe.yaml", PriceConstruction::SampledCurveCartesianProduct,
            {{
        {"model.mean_reversion_x", {0.005, pg::BumpScale::relative}},
        {"model.volatility_x", {0.005, pg::BumpScale::relative}},
        {"model.mean_reversion_y", {0.005, pg::BumpScale::relative}},
        {"model.volatility_y", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"curve.beta0", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta1", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta2", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta3", {0.0005, pg::BumpScale::absolute}},
        {"curve.tau1", {0.005, pg::BumpScale::relative}},
        {"curve.tau2", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true
        };
        recipe.option_side = "call";
        recipe.product_quote_mode = "ratio_to_initial_forward_bond_price";
        datasets::price_gradients::require_product_quote_convention(recipe);
        recipe.barrier_grid_monitoring = true;
        recipe.curve_input = "datasets/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::g2_plus_plus::load_models(recipe.model_input);
        const auto curves = curve::svensson::load_curves(recipe.curve_input);
        const auto products =
            product::load_zero_coupon_bond_up_and_outs(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_g2_plus_plus_svensson_zero_coupon_bond_up_and_out_price_gradients(
                    model_rows, curve_rows, product_rows,
                    construction, time, configuration
                );
        };
        const auto launch = [](
            const auto& plan,
            auto inputs,
            auto stencils,
            const auto& configuration,
            auto outputs
        ) {
            model_namespace::
                launch_g2_plus_plus_svensson_zero_coupon_bond_up_and_out_price_gradients_cuda(
                    plan, inputs, stencils, configuration, outputs
                );
        };
        return offline::pricing::price_gradients::execute_curve_dataset<
            true,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::fixed_income_mc,
                "g2_plus_plus",
                "zero_coupon_bond_up_and_out",
                "svensson",
            },
            11668829280216809472ULL,
            models,
            curves, products,
            prepare,
            launch,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model_namespace::
                prepare_zero_coupon_bond_up_and_out_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
