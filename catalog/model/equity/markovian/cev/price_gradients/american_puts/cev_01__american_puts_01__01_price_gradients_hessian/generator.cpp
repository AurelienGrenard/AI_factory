// Generated cev American selected full Hessian with frozen exercise.
#include "model/equity/markovian/cev/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/mixed_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/american_option/american_options_01.json", "datasets/model/equity/markovian/cev/price_gradients/american_puts/cev_01__american_puts_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/cev/price_gradients/american_puts/cev_01__american_puts_01__01_price_gradients_hessian/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/equity/markovian/cev/price_gradients/american_puts/cev_01__american_puts_01__01_price_gradients_hessian.json", "catalog/model/equity/markovian/cev/prices/american_puts/cev_01__american_puts_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.sigma", {0.005, pg::BumpScale::relative}},
        {"model.beta", {0.002, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, false,
            pg::SensitivityOrders::first_and_second
        };
        recipe.sensitivity_request = pg::SensitivityRequest::full_hessian();
        const auto models =
            model::equity::cev::load_models(recipe.model_input);
        const auto products =
            product::load_american_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& selected,
            pg::SensitivityRequest request
        ) {
            return model::equity::cev::
                prepare_cev_american_option_sensitivities(
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
                    offline::cuda_tuning::PricingFamily::equity_lsm,
                    "cev",
                    "american_option",
                    "",
                },
                11668826853560287232ULL,
                models,
                products,
                prepare,
                model::equity::cev::
                    cev_american_option_mixed_node_graph_workspace_bytes<
                        OptionSide::put>,
                model::equity::cev::
                    launch_cev_american_option_mixed_node_graph_sensitivities_cuda<
                        OptionSide::put>,
                offline::cuda_tuning::kProductionPathsPerPrice,
                model::equity::cev::
                    prepare_american_option_diagonal_sensitivity_stencils_cuda
            );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
