// Generated ornstein_uhlenbeck European-swaption gradient and diagonal-Hessian recipe.
#include "model/fixed_income/ornstein_uhlenbeck/product/european_swaption_price_gradients.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::ornstein_uhlenbeck;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01.json", "datasets/product/european_swaption/european_swaptions_01.json", "datasets/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/european_payer_swaptions/ornstein_uhlenbeck_01__european_payer_swaptions_01__01_price_sensitivities_diagonal.json", "catalog/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/european_payer_swaptions/ornstein_uhlenbeck_01__european_payer_swaptions_01__01_price_sensitivities_diagonal/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/ornstein_uhlenbeck/price_sensitivities/european_payer_swaptions/ornstein_uhlenbeck_01__european_payer_swaptions_01__01_price_sensitivities_diagonal.json", "catalog/model/fixed_income/ornstein_uhlenbeck/prices/european_payer_swaptions/ornstein_uhlenbeck_01__european_payer_swaptions_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true,
            pg::SensitivityOrders::first_and_second
        };
        const auto models =
            model::fixed_income::ornstein_uhlenbeck::load_models(recipe.model_input);
        const auto product_dataset =
            product::load_european_swaptions(recipe.product_input);
        const auto& products = product_dataset.products;
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_ornstein_uhlenbeck_european_swaption_sensitivities(
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
                launch_ornstein_uhlenbeck_european_swaption_diagonal_sensitivities_cuda<
                    SwaptionSide::payer,
                    pg::SensitivityOrders::first_and_second
                >(
                    plan, inputs, stencils, configuration, outputs
                    ,
                    closed_form::WorkDistribution::cooperative
                );
        };
        return offline::pricing::price_gradients::execute_dataset<
            false,
            pg::SensitivityOrders::first_and_second
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::jamshidian,
                "ornstein_uhlenbeck",
                "european_swaption",
                "",
            },
            0ULL,
            models,
            products,
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
