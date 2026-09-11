// Generated G2 European receiver swaption MC recipe.
#include "model/fixed_income/g2/product/european_swaption.cuh"
#include "model/fixed_income/g2/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/european_swaption_monte_carlo_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::g2;
    const datasets::EuropeanSwaptionMonteCarloRecipe recipe{
        "datasets/model/fixed_income/g2/parameters/g2_01.json",
        "",
        "datasets/product/european_swaption/european_swaptions_01.json",
        "datasets/model/fixed_income/g2/prices/european_receiver_swaptions/g2_01__european_receiver_swaptions_01__01.json",
        "catalog/model/fixed_income/g2/prices/european_receiver_swaptions/g2_01__european_receiver_swaptions_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/g2/prices/european_receiver_swaptions/g2_01__european_receiver_swaptions_01__01.json",
        11668829164252692480ULL,
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::fixed_income_mc, "g2", "european_swaption", ""},
    };
    const auto models = rates::load_models(recipe.model_dataset_path);
    const auto product_dataset = product::load_european_swaptions(recipe.product_dataset_path);
    const auto& products = product_dataset.products;
    const auto count = price_row_count(models.size(), products.size(), PriceConstruction::Aligned);
    datasets::generate_european_swaption_monte_carlo_prices(
        recipe, offline::cuda::inputs(models, products), count,
        [&](auto& execution, std::size_t offset, std::size_t batch, std::size_t paths,
            const offline::cuda_tuning::PricingLaunchPlan& plan) {
            rates::launch_g2_european_swaption_cuda<SwaptionSide::receiver>(
                execution.template input<0U>(), models.size(),
                products.data(), execution.template input<1U>(), products.size(),
                PriceConstruction::Aligned, count, offset, batch, paths, 1.0f / 252.0f,
                plan.profile.threads_per_block, plan.blocks_for(batch), recipe.seed, execution.prices(), execution.standard_errors()
            );
        }
    );
}
