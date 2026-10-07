// Generated G2 grid-monitored zero-coupon bond barrier call recipe.
#include "model/fixed_income/g2/product/zero_coupon_bond_up_and_out.cuh"
#include "model/fixed_income/g2/dataset.hpp"
#include "product/zero_coupon_bond_up_and_out/dataset.hpp"
#include "tools/pricing/zero_coupon_bond_up_and_out_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::g2;
    const datasets::ZeroCouponBondUpAndOutRecipe recipe{
        "datasets/model/fixed_income/g2/parameters/g2_01.json",
        "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01.json",
        "datasets/model/fixed_income/g2/prices/zero_coupon_bond_up_and_out_calls/g2_01__zero_coupon_bond_up_and_out_calls_01__01.json", "catalog/model/fixed_income/g2/prices/zero_coupon_bond_up_and_out_calls/g2_01__zero_coupon_bond_up_and_out_calls_01__01/generation.yaml", "https://datasets.ai-factory.example/v2/model/fixed_income/g2/prices/zero_coupon_bond_up_and_out_calls/g2_01__zero_coupon_bond_up_and_out_calls_01__01.json",
        11668829267331907584ULL, ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::fixed_income_mc, "g2", "zero_coupon_bond_up_and_out", ""}, PriceConstruction::Aligned,
        65'536U, 1U, {}, "exact joint Gaussian factor and stochastic rate-integral transitions",
    };
    const auto models = rates::load_models(recipe.model_dataset_path);
    const auto products = product::load_zero_coupon_bond_up_and_outs(recipe.product_dataset_path);
    const auto count = price_row_count(models.size(), products.size(), recipe.construction);
    datasets::generate_zero_coupon_bond_up_and_out_prices(
        recipe, offline::cuda::inputs(models, products), count,
        [&](auto& execution, std::size_t offset, std::size_t batch, std::size_t paths,
            const offline::cuda_tuning::PricingLaunchPlan& plan) {
            rates::launch_g2_zero_coupon_bond_up_and_out_cuda(
                execution.template input<0U>(), models.size(), products.data(),
                execution.template input<1U>(), products.size(), recipe.construction,
                count, offset, batch, paths, recipe.simulation_steps_per_day,
                plan.profile.threads_per_block, plan.blocks_for(batch),
                recipe.seed, execution.prices(), execution.standard_errors());
        });
}
