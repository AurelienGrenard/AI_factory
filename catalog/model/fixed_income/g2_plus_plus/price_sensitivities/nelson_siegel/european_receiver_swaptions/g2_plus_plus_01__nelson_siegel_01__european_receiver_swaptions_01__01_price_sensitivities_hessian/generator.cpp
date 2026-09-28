// Generated g2_plus_plus European-swaption selected full-Hessian recipe.
#include "model/fixed_income/g2_plus_plus/product/nelson_siegel/european_swaption_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/nelson_siegel/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2_plus_plus::nelson_siegel;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json", "datasets/product/european_swaption/european_swaptions_01.json", "datasets/model/fixed_income/g2_plus_plus/price_sensitivities/nelson_siegel/european_receiver_swaptions/g2_plus_plus_01__nelson_siegel_01__european_receiver_swaptions_01__01_price_sensitivities_hessian.json", "catalog/model/fixed_income/g2_plus_plus/price_sensitivities/nelson_siegel/european_receiver_swaptions/g2_plus_plus_01__nelson_siegel_01__european_receiver_swaptions_01__01_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/fixed_income/g2_plus_plus/price_sensitivities/nelson_siegel/european_receiver_swaptions/g2_plus_plus_01__nelson_siegel_01__european_receiver_swaptions_01__01_price_sensitivities_hessian.json", "catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/european_receiver_swaptions/g2_plus_plus_01__nelson_siegel_01__european_receiver_swaptions_01__01/recipe.yaml", PriceConstruction::Aligned,
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
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        recipe.curve_input = "datasets/curve/nelson_siegel/nelson_siegel_01.json";
        const auto models =
            model::fixed_income::g2_plus_plus::load_models(recipe.model_input);
        const auto curves = curve::nelson_siegel::load_curves(recipe.curve_input);
        const auto product_dataset =
            product::load_european_swaptions(recipe.product_input);
        const auto& products = product_dataset.products;
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_g2_plus_plus_nelson_siegel_european_swaption_sensitivities(
                    model_rows,
                    curve_rows, product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        const auto launch = [](
            const auto& plan,
            auto inputs,
            auto stencils,
            auto mixed_stencils,
            const auto& configuration,
            auto outputs,
            auto mixed_outputs,
            void* workspace,
            std::size_t workspace_bytes
        ) {
            return model_namespace::
                launch_g2_plus_plus_nelson_siegel_european_swaption_mixed_node_graph_sensitivities_cuda<
                    SwaptionSide::receiver>(
                        plan,
                        inputs,
                        stencils,
                        mixed_stencils,
                        configuration,
                        outputs,
                        mixed_outputs,
                        workspace,
                        workspace_bytes
                    );
        };
        return offline::pricing::price_gradients::
            execute_curve_mixed_node_graph_dataset<true>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::fixed_income_mc,
                    "g2_plus_plus",
                    "european_swaption",
                    "nelson_siegel",
                },
                11668829172842627072ULL,
                models,
                curves, products,
                prepare,
                model_namespace::
                    g2_plus_plus_nelson_siegel_european_swaption_mixed_node_graph_workspace_bytes<
                        SwaptionSide::receiver>,
                launch,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model_namespace::
                    prepare_european_swaption_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
