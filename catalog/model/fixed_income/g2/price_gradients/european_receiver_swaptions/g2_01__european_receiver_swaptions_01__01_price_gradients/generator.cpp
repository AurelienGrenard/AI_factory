// Generated g2 European-swaption selected-gradient recipe.
#include "model/fixed_income/g2/product/european_swaption_price_gradients.cuh"
#include "model/fixed_income/g2/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/model/fixed_income/g2/parameters/g2_01.json", "datasets/product/european_swaption/european_swaptions_01.json", "datasets/model/fixed_income/g2/price_gradients/european_receiver_swaptions/g2_01__european_receiver_swaptions_01__01_price_gradients.json", "catalog/model/fixed_income/g2/price_gradients/european_receiver_swaptions/g2_01__european_receiver_swaptions_01__01_price_gradients/generation.yaml",
            "https://datasets.ai-factory.example/v2/model/fixed_income/g2/price_gradients/european_receiver_swaptions/g2_01__european_receiver_swaptions_01__01_price_gradients.json", "catalog/model/fixed_income/g2/prices/european_receiver_swaptions/g2_01__european_receiver_swaptions_01__01/recipe.yaml", PriceConstruction::Aligned,
            {{
        {"model.mean_reversion_x", {0.005, pg::BumpScale::relative}},
        {"model.volatility_x", {0.005, pg::BumpScale::relative}},
        {"model.mean_reversion_y", {0.005, pg::BumpScale::relative}},
        {"model.volatility_y", {0.005, pg::BumpScale::relative}},
        {"model.correlation", {0.002, pg::BumpScale::absolute}},
        {"model.initial_state_x", {0.0005, pg::BumpScale::absolute}},
        {"model.initial_state_y", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.0005, pg::BumpScale::absolute}}
            }}, {1.0f / 504.0f, 2U}, true
        };
        const auto models =
            model::fixed_income::g2::load_models(recipe.model_input);
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
                prepare_g2_european_swaption_price_gradients(
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
                launch_g2_european_swaption_price_gradients_cuda<
                    SwaptionSide::receiver
                >(
                    plan, inputs, stencils, configuration, outputs
                    
                );
        };
        return offline::pricing::price_gradients::execute_dataset<
            true,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::fixed_income_mc,
                "g2",
                "european_swaption",
                "",
            },
            11668829164252692480ULL,
            models,
            products,
            prepare,
            launch,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model_namespace::
                prepare_european_swaption_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
