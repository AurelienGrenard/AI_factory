// Generated g2_plus_plus zero_coupon_bond_option gradient and diagonal-Hessian recipe.
#include "model/fixed_income/g2_plus_plus/product/nelson_siegel/zero_coupon_bond_option_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/nelson_siegel/dataset.hpp"
#include "product/zero_coupon_bond_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2_plus_plus::nelson_siegel;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json", "datasets/product/zero_coupon_bond_option/zero_coupon_bond_options_01.json", "datasets/model/fixed_income/g2_plus_plus/price_sensitivities/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01_cartesian_price_sensitivities_diagonal.json", "catalog/model/fixed_income/g2_plus_plus/price_sensitivities/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01_cartesian_price_sensitivities_diagonal/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/g2_plus_plus/price_sensitivities/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01_cartesian_price_sensitivities_diagonal.json", "catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.mean_reversion_x", {0.005, pg::BumpScale::relative}},
        {"model.volatility_x", {0.005, pg::BumpScale::relative}},
        {"model.mean_reversion_y", {0.005, pg::BumpScale::relative}},
        {"model.volatility_y", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"curve.beta0", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta1", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta2", {0.0005, pg::BumpScale::absolute}},
        {"curve.tau", {0.005, pg::BumpScale::relative}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.curve_input = "datasets/curve/nelson_siegel/nelson_siegel_01.json";
        const auto models =
            model::fixed_income::g2_plus_plus::load_models(recipe.model_input);
        const auto curves = curve::nelson_siegel::load_curves(recipe.curve_input);
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
                prepare_g2_plus_plus_nelson_siegel_zero_coupon_bond_option_sensitivities(
                    model_rows, curve_rows, product_rows,
                    construction, time, configuration,
                    {pg::SensitivityOrders::first_and_second}
                );
        };
        return offline::pricing::price_gradients::execute_curve_dataset<
            false,
            pg::SensitivityOrders::first_and_second
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::closed_form,
                "g2_plus_plus",
                "zero_coupon_bond_option",
                "nelson_siegel",
            },
            0ULL,
            models,
            curves, products,
            prepare,
            model_namespace::
                launch_g2_plus_plus_nelson_siegel_zero_coupon_bond_option_diagonal_sensitivities_cuda<
                    OptionSide::call,
                    pg::SensitivityOrders::first_and_second
                >,
            0U,
            model_namespace::
                prepare_zero_coupon_bond_option_diagonal_sensitivity_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
