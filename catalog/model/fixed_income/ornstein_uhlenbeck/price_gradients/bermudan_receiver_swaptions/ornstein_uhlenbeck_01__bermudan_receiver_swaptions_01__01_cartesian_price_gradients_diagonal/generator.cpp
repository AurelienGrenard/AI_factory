// Generated ornstein_uhlenbeck Bermudan-swaption gradient and diagonal-Hessian recipe.
#include "model/fixed_income/ornstein_uhlenbeck/product/bermudan_swaption_price_gradients.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "product/bermudan_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::ornstein_uhlenbeck;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01.json", "datasets/product/bermudan_swaption/bermudan_swaptions_01.json", "datasets/model/fixed_income/ornstein_uhlenbeck/price_gradients/bermudan_receiver_swaptions/ornstein_uhlenbeck_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/fixed_income/ornstein_uhlenbeck/price_gradients/bermudan_receiver_swaptions/ornstein_uhlenbeck_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradients_diagonal/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/fixed_income/ornstein_uhlenbeck/price_gradients/bermudan_receiver_swaptions/ornstein_uhlenbeck_01__bermudan_receiver_swaptions_01__01_cartesian_price_gradients_diagonal.json", "catalog/model/fixed_income/ornstein_uhlenbeck/prices/bermudan_receiver_swaptions/ornstein_uhlenbeck_01__bermudan_receiver_swaptions_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.notional", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}},
        {"product.accrual_fraction", {0.005, pg::BumpScale::relative}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        const auto models =
            model::fixed_income::ornstein_uhlenbeck::load_models(recipe.model_input);
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
                prepare_ornstein_uhlenbeck_bermudan_swaption_sensitivities(
                    model_rows, product_rows,
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
                launch_ornstein_uhlenbeck_bermudan_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::receiver,
                    pg::SensitivityOrders::first_and_second
                >(plan, inputs, stencils, configuration, outputs);
        };
        return offline::pricing::price_gradients::execute_dataset<
            true,
            pg::SensitivityOrders::first_and_second
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::gaussian_rate_lsm,
                "ornstein_uhlenbeck",
                "bermudan_swaption",
                "",
            },
            11668829129892954112ULL,
            models,
            products,
            prepare,
            launch,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model_namespace::
                prepare_bermudan_swaption_diagonal_sensitivity_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
