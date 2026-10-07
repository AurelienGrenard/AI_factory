// Generated CIR++/Flat grid-monitored zero-coupon bond barrier recipe.
#include "model/fixed_income/cir_plus_plus/product/flat/zero_coupon_bond_up_and_out.cuh"
#include "model/fixed_income/cir_plus_plus/dataset.hpp"
#include "curve/flat/dataset.hpp"
#include "product/zero_coupon_bond_up_and_out/dataset.hpp"
#include "tools/pricing/zero_coupon_bond_up_and_out_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::cir_plus_plus;
    namespace fitted = rates::flat;
    const datasets::ZeroCouponBondUpAndOutRecipe recipe{
        "datasets/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json",
        "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01.json",
        "datasets/model/fixed_income/cir_plus_plus/prices/flat/zero_coupon_bond_up_and_out_calls/cir_plus_plus_01__flat_01__zero_coupon_bond_up_and_out_calls_01__01.json", "catalog/model/fixed_income/cir_plus_plus/prices/flat/zero_coupon_bond_up_and_out_calls/cir_plus_plus_01__flat_01__zero_coupon_bond_up_and_out_calls_01__01/generation.yaml", "https://datasets.ai-factory.example/v2/model/fixed_income/cir_plus_plus/prices/flat/zero_coupon_bond_up_and_out_calls/cir_plus_plus_01__flat_01__zero_coupon_bond_up_and_out_calls_01__01.json",
        11668829254447005696ULL, ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::fixed_income_mc, "cir_plus_plus", "zero_coupon_bond_up_and_out", "flat"}, PriceConstruction::Aligned,
        1'048'576U, 1U, "datasets/curve/flat/flat_01.json",
        "exact noncentral chi-square factor transitions; trapezoidal stochastic rate integral",
    };
    const auto models = rates::load_models(recipe.model_dataset_path);
    const auto curves = curve::flat::load_curves(recipe.curve_dataset_path);
    const auto products = product::load_zero_coupon_bond_up_and_outs(recipe.product_dataset_path);
    const auto count = price_row_count(models.size(), curves.size(), products.size(), recipe.construction);
    datasets::generate_zero_coupon_bond_up_and_out_prices(
        recipe, offline::cuda::inputs(models, curves, products), count,
        [&](auto& execution, std::size_t offset, std::size_t batch, std::size_t paths,
            const offline::cuda_tuning::PricingLaunchPlan& plan) {
            fitted::launch_cir_plus_plus_flat_zero_coupon_bond_up_and_out_cuda(
                execution.template input<0U>(), models.size(),
                execution.template input<1U>(), curves.size(), products.data(),
                execution.template input<2U>(), products.size(), recipe.construction,
                count, offset, batch, paths, recipe.simulation_steps_per_day,
                plan.profile.threads_per_block, plan.blocks_for(batch),
                recipe.seed, execution.prices(), execution.standard_errors());
        });
}
