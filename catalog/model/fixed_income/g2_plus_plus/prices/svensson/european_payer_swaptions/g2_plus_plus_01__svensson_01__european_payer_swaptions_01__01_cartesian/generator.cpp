// Generated Svensson-fitted G2++ European payer swaption MC recipe.
#include "model/fixed_income/g2_plus_plus/product/svensson/european_swaption.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/european_swaption_monte_carlo_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::g2_plus_plus;
    const datasets::EuropeanSwaptionMonteCarloRecipe recipe{
        "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json",
        "datasets/curve/svensson/svensson_01.json",
        "datasets/product/european_swaption/european_swaptions_01.json",
        "datasets/model/fixed_income/g2_plus_plus/prices/svensson/european_payer_swaptions/g2_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian.json",
        "catalog/model/fixed_income/g2_plus_plus/prices/svensson/european_payer_swaptions/g2_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/g2_plus_plus/prices/svensson/european_payer_swaptions/g2_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian.json",
        11668829177137594368ULL,
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::fixed_income_mc, "g2_plus_plus", "european_swaption", "svensson"},
        PriceConstruction::CartesianProduct,
    };
    const auto models = rates::load_models(recipe.model_dataset_path);
    const auto curves = curve::svensson::load_curves(recipe.curve_dataset_path);
    const auto product_dataset = product::load_european_swaptions(recipe.product_dataset_path);
    const auto& products = product_dataset.products;
    const auto count = price_row_count(models.size(), curves.size(), products.size(), recipe.construction);
    datasets::generate_european_swaption_monte_carlo_prices(
        recipe, offline::cuda::inputs(models, curves, products), count,
        [&](auto& execution, std::size_t offset, std::size_t batch, std::size_t paths,
            const offline::cuda_tuning::PricingLaunchPlan& plan) {
            rates::svensson::launch_g2_plus_plus_svensson_european_swaption_cuda<SwaptionSide::payer>(
                execution.template input<0U>(), models.size(),
                execution.template input<1U>(), curves.size(),
                products.data(), execution.template input<2U>(), products.size(),
                recipe.construction, count, offset, batch, paths, 1.0f / 252.0f,
                plan.profile.threads_per_block, plan.blocks_for(batch), recipe.seed, execution.prices(), execution.standard_errors()
            );
        }
    );
}
