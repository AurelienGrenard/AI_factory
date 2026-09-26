// Generated schobel_zhu digital_option selected full-Hessian recipe.
#include "model/equity/markovian/schobel_zhu/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/markovian/schobel_zhu/price_gradients/digital_puts/schobel_zhu_01__digital_puts_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/schobel_zhu/price_gradients/digital_puts/schobel_zhu_01__digital_puts_01__01_cartesian_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/price_gradients/digital_puts/schobel_zhu_01__digital_puts_01__01_cartesian_price_gradients_hessian.json", "catalog/model/equity/markovian/schobel_zhu/prices/digital_puts/schobel_zhu_01__digital_puts_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_volatility", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}},
        {"product.cash_payoff", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::schobel_zhu::load_models(recipe.model_input);
        const auto products = product::load_digital_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::schobel_zhu::
                prepare_schobel_zhu_digital_option_sensitivities(
                    model_rows,
                    product_rows,
                    construction,
                    time,
                    selected,
                    std::move(request)
                );
        };
        return offline::pricing::price_gradients::
            execute_mixed_node_graph_dataset<true>(
                recipe,
                {
                    offline::cuda_tuning::PricingFamily::equity_step_mc,
                    "schobel_zhu",
                    "digital_option",
                    "",
                },
                11668827862877601792ULL,
                models,
                products,
                prepare,
                model::equity::schobel_zhu::
                    schobel_zhu_digital_option_mixed_node_graph_workspace_bytes<OptionSide::put>,
                model::equity::schobel_zhu::
                    launch_schobel_zhu_digital_option_mixed_node_graph_sensitivities_cuda<OptionSide::put>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::schobel_zhu::
                    prepare_digital_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
