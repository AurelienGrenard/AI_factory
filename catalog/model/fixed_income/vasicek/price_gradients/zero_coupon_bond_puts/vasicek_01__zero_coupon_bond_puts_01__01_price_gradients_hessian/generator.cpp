// Generated vasicek zero_coupon_bond_option selected full-Hessian recipe.
#include "model/fixed_income/vasicek/product/zero_coupon_bond_option_price_gradients.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"
#include "product/zero_coupon_bond_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::vasicek;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/vasicek/parameters/vasicek_01.json", "datasets/product/zero_coupon_bond_option/zero_coupon_bond_options_01.json", "datasets/model/fixed_income/vasicek/price_gradients/zero_coupon_bond_puts/vasicek_01__zero_coupon_bond_puts_01__01_price_gradients_hessian.json", "catalog/model/fixed_income/vasicek/price_gradients/zero_coupon_bond_puts/vasicek_01__zero_coupon_bond_puts_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/vasicek/price_gradients/zero_coupon_bond_puts/vasicek_01__zero_coupon_bond_puts_01__01_price_gradients_hessian.json", "catalog/model/fixed_income/vasicek/prices/zero_coupon_bond_puts/vasicek_01__zero_coupon_bond_puts_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_term_mean", {0.0005, pg::BumpScale::absolute}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::fixed_income::vasicek::load_models(recipe.model_input);
        const auto products =
            product::load_zero_coupon_bond_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_vasicek_zero_coupon_bond_option_sensitivities(
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
                    "vasicek",
                    "zero_coupon_bond_option",
                    "",
                },
                0ULL,
                models,
                products,
                prepare,
                model_namespace::
                    vasicek_zero_coupon_bond_option_mixed_node_graph_workspace_bytes<
                        OptionSide::put>,
                model_namespace::
                    launch_vasicek_zero_coupon_bond_option_mixed_node_graph_sensitivities_cuda<
                        OptionSide::put>,
                0U,
                model_namespace::
                    prepare_zero_coupon_bond_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
