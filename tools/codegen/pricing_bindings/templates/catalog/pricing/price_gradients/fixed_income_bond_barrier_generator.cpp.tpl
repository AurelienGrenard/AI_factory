// Generated ${model} zero-coupon bond barrier call first-order gradients.
#include "model/fixed_income/${model}/product/${curve_prefix}zero_coupon_bond_up_and_out_price_gradients.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
${curve_dataset_include}#include "product/zero_coupon_bond_up_and_out/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::${binding_namespace};
    try {
        datasets::price_gradients::Recipe recipe{
            "$model_input", "$product_input", "$dataset", "$catalog",
            "$source_recipe", PriceConstruction::$construction,
            {{
        $selections
            }}, {1.0f / 504.0f, 2U}, true
        };$option_side_assignment
        recipe.barrier_grid_monitoring = true;
${curve_recipe_assignment}        const auto models =
            model::fixed_income::${model}::load_models(recipe.model_input);
${curve_load}        const auto products =
            product::load_zero_coupon_bond_up_and_outs(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            ${curve_lambda_argument}const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_${function_prefix}_zero_coupon_bond_up_and_out_price_gradients(
                    model_rows, ${curve_prepare_argument}product_rows,
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
                launch_${function_prefix}_zero_coupon_bond_up_and_out_price_gradients_cuda(
                    plan, inputs, stencils, configuration, outputs
                );
        };
        return offline::pricing::price_gradients::${execute_function}<
            true,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::fixed_income_mc,
                "$model",
                "zero_coupon_bond_up_and_out",
                "$curve",
            },
            ${seed}ULL,
            models,
            ${execute_curve_argument}products,
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
