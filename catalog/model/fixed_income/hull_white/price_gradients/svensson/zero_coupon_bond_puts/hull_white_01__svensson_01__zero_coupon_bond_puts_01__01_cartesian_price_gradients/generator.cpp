// Generated hull_white zero_coupon_bond_option selected-gradient recipe.
#include "model/fixed_income/hull_white/product/svensson/zero_coupon_bond_option_price_gradients.cuh"
#include "model/fixed_income/hull_white/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/zero_coupon_bond_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::hull_white::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/hull_white/parameters/hull_white_01.json", "datasets/product/zero_coupon_bond_option/zero_coupon_bond_options_01.json", "datasets/model/fixed_income/hull_white/price_gradients/svensson/zero_coupon_bond_puts/hull_white_01__svensson_01__zero_coupon_bond_puts_01__01_cartesian_price_gradients.json", "catalog/model/fixed_income/hull_white/price_gradients/svensson/zero_coupon_bond_puts/hull_white_01__svensson_01__zero_coupon_bond_puts_01__01_cartesian_price_gradients/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/hull_white/price_gradients/svensson/zero_coupon_bond_puts/hull_white_01__svensson_01__zero_coupon_bond_puts_01__01_cartesian_price_gradients.json", "catalog/model/fixed_income/hull_white/prices/svensson/zero_coupon_bond_puts/hull_white_01__svensson_01__zero_coupon_bond_puts_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
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
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true
        };
        recipe.curve_input = "datasets/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::hull_white::load_models(recipe.model_input);
        const auto curves = curve::svensson::load_curves(recipe.curve_input);
        const auto products =
            product::load_zero_coupon_bond_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_hull_white_svensson_zero_coupon_bond_option_price_gradients(
                    model_rows, curve_rows, product_rows,
                    construction, time, configuration
                );
        };
        return offline::pricing::price_gradients::execute_curve_dataset<
            false,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::closed_form,
                "hull_white",
                "zero_coupon_bond_option",
                "svensson",
            },
            0ULL,
            models,
            curves, products,
            prepare,
            model_namespace::
                launch_hull_white_svensson_zero_coupon_bond_option_price_gradients_cuda<
                    OptionSide::put
                >,
            0U,
            model_namespace::
                prepare_zero_coupon_bond_option_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
