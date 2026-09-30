// Generated cir Bermudan-swaption gradient and diagonal-Hessian recipe.
#include "model/fixed_income/cir/product/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::cir;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/cir/parameters/cir_01.json", "datasets/product/bermudan_swaption/bermudan_swaptions_01.json", "datasets/model/fixed_income/cir/price_gradients/bermudan_receiver_swaptions/cir_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradient_diagonal_hessian.json", "catalog/model/fixed_income/cir/price_gradients/bermudan_receiver_swaptions/cir_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradient_diagonal_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/fixed_income/cir/price_gradients/bermudan_receiver_swaptions/cir_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradient_diagonal_hessian.json", "catalog/model/fixed_income/cir/prices/bermudan_receiver_swaptions/cir_01__bermudan_receiver_swaptions_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_term_mean", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_strategy =
            offline::pricing::price_gradients::
                sensitivity_strategy_from_arguments(argc, argv);
        const auto models =
            model::fixed_income::cir::load_models(recipe.model_input);
        const auto products =
            product::load_bermudan_swaptions(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_cir_bermudan_swaption_sensitivities(
                    model_rows, product_rows,
                    construction, time, configuration,
                    {pg::SensitivityOrders::first_and_second}
                );
        };
        return offline::pricing::price_gradients::
            execute_node_graph_dataset<
            pg::SensitivityOrders::first_and_second
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::terminal_forward_lsm,
                "cir",
                "bermudan_swaption",
                "",
            },
            11668829043993608192ULL,
            models,
            products,
            prepare,
            model_namespace::
                launch_cir_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::receiver,
                    pg::SensitivityOrders::first_and_second
                >,
            model_namespace::
                cir_bermudan_swaption_node_graph_workspace_bytes<
                    SwaptionSide::receiver,
                    pg::SensitivityOrders::first_and_second
                >,
            model_namespace::
                launch_cir_bermudan_swaption_node_graph_sensitivities_cuda<
                    SwaptionSide::receiver,
                    pg::SensitivityOrders::first_and_second
                >,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model_namespace::
                prepare_bermudan_swaption_diagonal_sensitivity_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
