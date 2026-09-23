// Generated ${model_display} European ${swaption_side} swaption MC recipe.
#include "model/fixed_income/${model}/product/european_swaption.cuh"
#include "model/fixed_income/${model}/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/european_swaption_monte_carlo_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::${model};
    const datasets::EuropeanSwaptionMonteCarloRecipe recipe{
        "datasets/model/fixed_income/${model}/parameters/${model}_01.json",
        "",
        "datasets/product/european_swaption/european_swaptions_01.json",
        "datasets/model/fixed_income/${model}/prices/${variant}/${model}_01__${variant}_01__01.json",
        "catalog/model/fixed_income/${model}/prices/${variant}/${model}_01__${variant}_01__01/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/${model}/prices/${variant}/${model}_01__${variant}_01__01.json",
        ${dynamics_seed}ULL,
        ${launch_identity},
        PriceConstruction::${construction},
    };
    const auto models = rates::load_models(recipe.model_dataset_path);
    const auto product_dataset = product::load_european_swaptions(recipe.product_dataset_path);
    const auto& products = product_dataset.products;
    const auto count = price_row_count(models.size(), products.size(), recipe.construction);
    datasets::generate_european_swaption_monte_carlo_prices(
        recipe, offline::cuda::inputs(models, products), count,
        [&](auto& execution, std::size_t offset, std::size_t batch, std::size_t paths,
            const offline::cuda_tuning::PricingLaunchPlan& plan) {
            rates::launch_${model}_european_swaption_cuda<SwaptionSide::${swaption_side}>(
                execution.template input<0U>(), models.size(),
                products.data(), execution.template input<1U>(), products.size(),
                recipe.construction, count, offset, batch, paths, 1.0f / 252.0f,
                plan.profile.threads_per_block, plan.blocks_for(batch), recipe.seed, execution.prices(), execution.standard_errors()
            );
        }
    );
}
