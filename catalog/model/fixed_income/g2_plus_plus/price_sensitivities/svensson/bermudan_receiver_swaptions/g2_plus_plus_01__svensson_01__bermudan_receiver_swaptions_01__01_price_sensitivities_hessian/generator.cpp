// Generated g2_plus_plus Bermudan selected full Hessian with frozen exercise.
#include "model/fixed_income/g2_plus_plus/product/svensson/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2_plus_plus::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json", "datasets/product/bermudan_swaption/bermudan_swaptions_01.json", "datasets/model/fixed_income/g2_plus_plus/price_sensitivities/svensson/bermudan_receiver_swaptions/g2_plus_plus_01__svensson_01__bermudan_receiver_swaptions_01__01_price_sensitivities_hessian.json", "catalog/model/fixed_income/g2_plus_plus/price_sensitivities/svensson/bermudan_receiver_swaptions/g2_plus_plus_01__svensson_01__bermudan_receiver_swaptions_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/fixed_income/g2_plus_plus/price_sensitivities/svensson/bermudan_receiver_swaptions/g2_plus_plus_01__svensson_01__bermudan_receiver_swaptions_01__01_price_sensitivities_hessian.json", "catalog/model/fixed_income/g2_plus_plus/prices/svensson/bermudan_receiver_swaptions/g2_plus_plus_01__svensson_01__bermudan_receiver_swaptions_01__01/recipe.yaml", PriceConstruction::Aligned,
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
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        recipe.curve_input = "datasets/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::g2_plus_plus::load_models(recipe.model_input);
        const auto curves = curve::svensson::load_curves(recipe.curve_input);
        const auto products =
            product::load_bermudan_swaptions(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_g2_plus_plus_svensson_bermudan_swaption_sensitivities(
                    model_rows,
                    curve_rows, product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        return offline::pricing::price_gradients::
            execute_curve_mixed_node_graph_dataset<true>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::gaussian_rate_lsm,
                    "g2_plus_plus",
                    "bermudan_swaption",
                    "svensson",
                },
                11668829086943281152ULL,
                models,
                curves, products,
                prepare,
                model_namespace::
                    g2_plus_plus_svensson_bermudan_swaption_mixed_node_graph_workspace_bytes<
                        SwaptionSide::receiver>,
                model_namespace::
                    launch_g2_plus_plus_svensson_bermudan_swaption_mixed_node_graph_sensitivities_cuda<
                        SwaptionSide::receiver>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model_namespace::
                    prepare_bermudan_swaption_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
