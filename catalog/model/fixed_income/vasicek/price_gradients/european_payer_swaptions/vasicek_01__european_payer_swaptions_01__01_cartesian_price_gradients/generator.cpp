// Generated vasicek European-swaption selected-gradient recipe.
#include "model/fixed_income/vasicek/product/european_swaption_price_gradients.cuh"
#include "model/fixed_income/vasicek/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::vasicek;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/vasicek/parameters/vasicek_01.json", "datasets/product/european_swaption/european_swaptions_01.json", "datasets/model/fixed_income/vasicek/price_gradients/european_payer_swaptions/vasicek_01__european_payer_swaptions_01__01_cartesian_price_gradients.json", "catalog/model/fixed_income/vasicek/price_gradients/european_payer_swaptions/vasicek_01__european_payer_swaptions_01__01_cartesian_price_gradients/generation.yaml",
            "https://datasets.ai-factory.example/v1/model/fixed_income/vasicek/price_gradients/european_payer_swaptions/vasicek_01__european_payer_swaptions_01__01_cartesian_price_gradients.json", "catalog/model/fixed_income/vasicek/prices/european_payer_swaptions/vasicek_01__european_payer_swaptions_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {{
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_term_mean", {0.0005, pg::BumpScale::absolute}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.initial_state", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true
        };
        const auto models =
            model::fixed_income::vasicek::load_models(recipe.model_input);
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
                prepare_vasicek_european_swaption_price_gradients(
                    model_rows, product_rows,
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
                launch_vasicek_european_swaption_price_gradients_cuda<
                    SwaptionSide::payer
                >(
                    plan, inputs, stencils, configuration, outputs
                    ,
                    closed_form::WorkDistribution::cooperative
                );
        };
        return offline::pricing::price_gradients::execute_dataset<
            false,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::jamshidian,
                "vasicek",
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
                prepare_european_swaption_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
