// Generated cir zero-coupon bond barrier call first-order gradients.
#include "model/fixed_income/cir/product/zero_coupon_bond_up_and_out_price_gradients.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "product/zero_coupon_bond_up_and_out/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::cir;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/cir/parameters/cir_01.json", "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01.json", "datasets/model/fixed_income/cir/price_gradients/zero_coupon_bond_up_and_out_calls/cir_01__zero_coupon_bond_up_and_out_calls_01__01_price_gradient.json", "catalog/model/fixed_income/cir/price_gradients/zero_coupon_bond_up_and_out_calls/cir_01__zero_coupon_bond_up_and_out_calls_01__01_price_gradient/generation.yaml",
            "catalog/model/fixed_income/cir/prices/zero_coupon_bond_up_and_out_calls/cir_01__zero_coupon_bond_up_and_out_calls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_term_mean", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.barrier", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true
        };
        recipe.option_side = "call";
        recipe.product_quote_mode = "ratio_to_initial_forward_bond_price";
        datasets::price_gradients::require_product_quote_convention(recipe);
        recipe.barrier_grid_monitoring = true;
        const auto models =
            model::fixed_income::cir::load_models(recipe.model_input);
        const auto products =
            product::load_zero_coupon_bond_up_and_outs(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_cir_zero_coupon_bond_up_and_out_price_gradients(
                    model_rows, product_rows,
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
                launch_cir_zero_coupon_bond_up_and_out_price_gradients_cuda(
                    plan, inputs, stencils, configuration, outputs
                );
        };
        return offline::pricing::price_gradients::execute_dataset<
            true,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::fixed_income_mc,
                "cir",
                "zero_coupon_bond_up_and_out",
                "",
            },
            11668829250152038400ULL,
            models,
            products,
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
