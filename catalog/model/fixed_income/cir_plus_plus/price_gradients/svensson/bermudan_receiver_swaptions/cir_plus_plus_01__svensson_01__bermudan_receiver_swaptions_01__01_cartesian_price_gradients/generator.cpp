// Generated cir_plus_plus Bermudan-swaption selected-gradient recipe.
#include "model/fixed_income/cir_plus_plus/product/svensson/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/cir_plus_plus/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::cir_plus_plus::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json", "datasets/product/bermudan_swaption/bermudan_swaptions_01.json", "datasets/model/fixed_income/cir_plus_plus/price_gradients/svensson/bermudan_receiver_swaptions/cir_plus_plus_01__svensson_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradients.json", "catalog/model/fixed_income/cir_plus_plus/price_gradients/svensson/bermudan_receiver_swaptions/cir_plus_plus_01__svensson_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradients/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/fixed_income/cir_plus_plus/price_gradients/svensson/bermudan_receiver_swaptions/cir_plus_plus_01__svensson_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradients.json", "catalog/model/fixed_income/cir_plus_plus/prices/svensson/bermudan_receiver_swaptions/cir_plus_plus_01__svensson_01__bermudan_receiver_swaptions_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
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
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true
        };
        recipe.curve_input = "datasets/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::cir_plus_plus::load_models(recipe.model_input);
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
                prepare_cir_plus_plus_svensson_bermudan_swaption_price_gradients(
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
                launch_cir_plus_plus_svensson_bermudan_swaption_price_gradients_cuda<
                    SwaptionSide::receiver
                >(plan, inputs, stencils, configuration, outputs);
        };
        return offline::pricing::price_gradients::execute_curve_dataset<
            true,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::terminal_forward_lsm,
                "cir_plus_plus",
                "bermudan_swaption",
                "svensson",
            },
            11668829198612430848ULL,
            models,
            curves, products,
            prepare,
            launch,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model_namespace::
                prepare_bermudan_swaption_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
