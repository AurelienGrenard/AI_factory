// Generated ornstein_uhlenbeck zero_coupon_bond_option selected-gradient recipe.
#include "model/fixed_income/ornstein_uhlenbeck/product/zero_coupon_bond_option_price_gradients.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "product/zero_coupon_bond_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::ornstein_uhlenbeck;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01.json", "datasets/product/zero_coupon_bond_option/zero_coupon_bond_options_01.json", "datasets/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/zero_coupon_bond_puts/ornstein_uhlenbeck_01__zero_coupon_bond_puts_01__01_price_sensitivities.json", "catalog/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/zero_coupon_bond_puts/ornstein_uhlenbeck_01__zero_coupon_bond_puts_01__01_price_sensitivities/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/zero_coupon_bond_puts/ornstein_uhlenbeck_01__zero_coupon_bond_puts_01__01_price_sensitivities.json", "catalog/model/fixed_income/ornstein_uhlenbeck/prices/zero_coupon_bond_puts/ornstein_uhlenbeck_01__zero_coupon_bond_puts_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true
        };
        const auto models =
            model::fixed_income::ornstein_uhlenbeck::load_models(recipe.model_input);
        const auto products =
            product::load_zero_coupon_bond_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_ornstein_uhlenbeck_zero_coupon_bond_option_price_gradients(
                    model_rows, product_rows,
                    construction, time, configuration
                );
        };
        return offline::pricing::price_gradients::execute_dataset<
            false,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::closed_form,
                "ornstein_uhlenbeck",
                "zero_coupon_bond_option",
                "",
            },
            0ULL,
            models,
            products,
            prepare,
            model_namespace::
                launch_ornstein_uhlenbeck_zero_coupon_bond_option_price_gradients_cuda<
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
