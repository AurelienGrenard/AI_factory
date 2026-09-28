// Generated cir European-swaption selected full-Hessian recipe.
#include "model/fixed_income/cir/product/european_swaption_price_gradients.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::cir;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/cir/parameters/cir_01.json", "datasets/product/european_swaption/european_swaptions_01.json", "datasets/model/fixed_income/cir/price_sensitivities/european_payer_swaptions/cir_01__european_payer_swaptions_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/fixed_income/cir/price_sensitivities/european_payer_swaptions/cir_01__european_payer_swaptions_01__01_cartesian_price_sensitivities_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/cir/price_sensitivities/european_payer_swaptions/cir_01__european_payer_swaptions_01__01_cartesian_price_sensitivities_hessian.json", "catalog/model/fixed_income/cir/prices/european_payer_swaptions/cir_01__european_payer_swaptions_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_term_mean", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::fixed_income::cir::load_models(recipe.model_input);
        const auto product_dataset =
            product::load_european_swaptions(recipe.product_input);
        const auto& products = product_dataset.products;
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model_namespace::
                prepare_cir_european_swaption_sensitivities(
                    model_rows,
                    product_rows,
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
                launch_cir_european_swaption_mixed_node_graph_sensitivities_cuda<
                    SwaptionSide::payer>(
                        plan,
                        inputs,
                        stencils,
                        mixed_stencils,
                        configuration,
                        outputs,
                        mixed_outputs,
                        workspace,
                        workspace_bytes,
                        closed_form::WorkDistribution::cooperative
                    );
        };
        return offline::pricing::price_gradients::
            execute_mixed_node_graph_dataset<false>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::jamshidian,
                    "cir",
                    "european_swaption",
                    "",
                },
                0ULL,
                models,
                products,
                prepare,
                model_namespace::
                    cir_european_swaption_mixed_node_graph_workspace_bytes<
                        SwaptionSide::payer>,
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
