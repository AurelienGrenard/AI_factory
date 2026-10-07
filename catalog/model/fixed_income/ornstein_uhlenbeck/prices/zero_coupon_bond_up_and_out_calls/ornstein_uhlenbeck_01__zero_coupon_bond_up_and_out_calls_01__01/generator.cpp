// Generated Ornstein-Uhlenbeck grid-monitored zero-coupon bond barrier call recipe.
#include "model/fixed_income/ornstein_uhlenbeck/product/zero_coupon_bond_up_and_out.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/dataset.hpp"
#include "product/zero_coupon_bond_up_and_out/dataset.hpp"
#include "tools/pricing/zero_coupon_bond_up_and_out_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::ornstein_uhlenbeck;
    const datasets::ZeroCouponBondUpAndOutRecipe recipe{
        "datasets/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01.json",
        "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01.json",
        "datasets/model/fixed_income/ornstein_uhlenbeck/prices/zero_coupon_bond_up_and_out_calls/ornstein_uhlenbeck_01__zero_coupon_bond_up_and_out_calls_01__01.json", "catalog/model/fixed_income/ornstein_uhlenbeck/prices/zero_coupon_bond_up_and_out_calls/ornstein_uhlenbeck_01__zero_coupon_bond_up_and_out_calls_01__01/generation.yaml", "https://datasets.ai-factory.example/v2/model/fixed_income/ornstein_uhlenbeck/prices/zero_coupon_bond_up_and_out_calls/ornstein_uhlenbeck_01__zero_coupon_bond_up_and_out_calls_01__01.json",
        11668829245857071104ULL, ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::fixed_income_mc, "ornstein_uhlenbeck", "zero_coupon_bond_up_and_out", ""}, PriceConstruction::Aligned,
        1'048'576U, 1U, {}, "exact joint Gaussian factor and stochastic rate-integral transitions",
    };
    const auto models = rates::load_models(recipe.model_dataset_path);
    const auto products = product::load_zero_coupon_bond_up_and_outs(recipe.product_dataset_path);
    const auto count = price_row_count(models.size(), products.size(), recipe.construction);
    datasets::generate_zero_coupon_bond_up_and_out_prices(
        recipe, offline::cuda::inputs(models, products), count,
        [&](auto& execution, std::size_t offset, std::size_t batch, std::size_t paths,
            const offline::cuda_tuning::PricingLaunchPlan& plan) {
            rates::launch_ornstein_uhlenbeck_zero_coupon_bond_up_and_out_cuda(
                execution.template input<0U>(), models.size(), products.data(),
                execution.template input<1U>(), products.size(), recipe.construction,
                count, offset, batch, paths, recipe.simulation_steps_per_day,
                plan.profile.threads_per_block, plan.blocks_for(batch),
                recipe.seed, execution.prices(), execution.standard_errors());
        });
}
