// Generated hull_white European-swaption selected-gradient recipe.
#include "model/fixed_income/hull_white/product/svensson/european_swaption_price_gradients.cuh"
#include "model/fixed_income/hull_white/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::hull_white::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/hull_white/parameters/hull_white_01.json", "datasets/product/european_swaption/european_swaptions_01.json", "datasets/model/fixed_income/hull_white/price_gradients/svensson/european_payer_swaptions/hull_white_01__svensson_01__european_payer_swaptions_01__01_cartesian_price_gradients.json", "catalog/model/fixed_income/hull_white/price_gradients/svensson/european_payer_swaptions/hull_white_01__svensson_01__european_payer_swaptions_01__01_cartesian_price_gradients/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/hull_white/price_gradients/svensson/european_payer_swaptions/hull_white_01__svensson_01__european_payer_swaptions_01__01_cartesian_price_gradients.json", "catalog/model/fixed_income/hull_white/prices/svensson/european_payer_swaptions/hull_white_01__svensson_01__european_payer_swaptions_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"curve.beta0", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta1", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta2", {0.0005, pg::BumpScale::absolute}},
        {"curve.beta3", {0.0005, pg::BumpScale::absolute}},
        {"curve.tau1", {0.005, pg::BumpScale::relative}},
        {"curve.tau2", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true
        };
        recipe.curve_input = "datasets/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::hull_white::load_models(recipe.model_input);
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
                prepare_hull_white_svensson_european_swaption_price_gradients(
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
            return model_namespace::
                launch_hull_white_svensson_european_swaption_price_gradients_cuda<
                    SwaptionSide::payer
                >(
                    plan, inputs, stencils, configuration, outputs
                    ,
                    closed_form::WorkDistribution::cooperative
                );
        };
        return offline::pricing::price_gradients::execute_curve_dataset<
            false,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::jamshidian,
                "hull_white",
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
                prepare_european_swaption_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
