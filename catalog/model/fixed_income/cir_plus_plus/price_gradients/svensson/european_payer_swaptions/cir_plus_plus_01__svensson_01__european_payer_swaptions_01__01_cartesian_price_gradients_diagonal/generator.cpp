// Generated cir_plus_plus European-swaption gradient and diagonal-Hessian recipe.
#include "model/fixed_income/cir_plus_plus/product/svensson/european_swaption_price_gradients.cuh"
#include "model/fixed_income/cir_plus_plus/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::cir_plus_plus::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json", "datasets/product/european_swaption/european_swaptions_01.json", "datasets/model/fixed_income/cir_plus_plus/price_gradients/svensson/european_payer_swaptions/cir_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/fixed_income/cir_plus_plus/price_gradients/svensson/european_payer_swaptions/cir_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian_price_gradients_diagonal/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/cir_plus_plus/price_gradients/svensson/european_payer_swaptions/cir_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/fixed_income/cir_plus_plus/prices/svensson/european_payer_swaptions/cir_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
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
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.curve_input = "datasets/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::cir_plus_plus::load_models(recipe.model_input);
        const auto curves = curve::svensson::load_curves(recipe.curve_input);
        const auto product_dataset =
            product::load_european_swaptions(recipe.product_input);
        const auto& products = product_dataset.products;
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_cir_plus_plus_svensson_european_swaption_sensitivities(
                    model_rows, curve_rows, product_rows,
                    construction, time, configuration,
                    {pg::SensitivityOrders::first_and_second}
                );
        };
        const auto launch = [](
            const auto& plan,
            auto inputs,
            auto stencils,
            const auto& configuration,
            auto outputs
        ) {
            return model_namespace::
                launch_cir_plus_plus_svensson_european_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(
                    plan, inputs, stencils, configuration, outputs
                    ,
                    closed_form::WorkDistribution::cooperative
                );
        };
        return offline::pricing::price_gradients::execute_curve_dataset<
            false,
            pg::SensitivityOrders::first_and_second
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::jamshidian,
                "cir_plus_plus",
                "european_swaption",
                "svensson",
            },
            0ULL,
            models,
            curves, products,
            prepare,
            launch,
            0U,
            model_namespace::
                prepare_european_swaption_diagonal_sensitivity_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
