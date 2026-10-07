// Generated G2++/Svensson grid-monitored zero-coupon bond barrier recipe.
#include "model/fixed_income/g2_plus_plus/product/svensson/zero_coupon_bond_up_and_out.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/zero_coupon_bond_up_and_out/dataset.hpp"
#include "tools/pricing/zero_coupon_bond_up_and_out_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::g2_plus_plus;
    namespace fitted = rates::svensson;
    const datasets::ZeroCouponBondUpAndOutRecipe recipe{
        "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json",
        "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01.json",
        "datasets/model/fixed_income/g2_plus_plus/prices/svensson/zero_coupon_bond_up_and_out_calls/g2_plus_plus_01__svensson_01__zero_coupon_bond_up_and_out_calls_01__01.json", "catalog/model/fixed_income/g2_plus_plus/prices/svensson/zero_coupon_bond_up_and_out_calls/g2_plus_plus_01__svensson_01__zero_coupon_bond_up_and_out_calls_01__01/generation.yaml", "https://datasets.ai-factory.example/v2/model/fixed_income/g2_plus_plus/prices/svensson/zero_coupon_bond_up_and_out_calls/g2_plus_plus_01__svensson_01__zero_coupon_bond_up_and_out_calls_01__01.json",
        11668829280216809472ULL, ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::fixed_income_mc, "g2_plus_plus", "zero_coupon_bond_up_and_out", "svensson"}, PriceConstruction::Aligned,
        1'048'576U, 1U, "datasets/curve/svensson/svensson_01.json",
        "exact joint Gaussian factor and stochastic rate-integral transitions",
    };
    const auto models = rates::load_models(recipe.model_dataset_path);
    const auto curves = curve::svensson::load_curves(recipe.curve_dataset_path);
    const auto products = product::load_zero_coupon_bond_up_and_outs(recipe.product_dataset_path);
    const auto count = price_row_count(models.size(), curves.size(), products.size(), recipe.construction);
    datasets::generate_zero_coupon_bond_up_and_out_prices(
        recipe, offline::cuda::inputs(models, curves, products), count,
        [&](auto& execution, std::size_t offset, std::size_t batch, std::size_t paths,
            const offline::cuda_tuning::PricingLaunchPlan& plan) {
            fitted::launch_g2_plus_plus_svensson_zero_coupon_bond_up_and_out_cuda(
                execution.template input<0U>(), models.size(),
                execution.template input<1U>(), curves.size(), products.data(),
                execution.template input<2U>(), products.size(), recipe.construction,
                count, offset, batch, paths, recipe.simulation_steps_per_day,
                plan.profile.threads_per_block, plan.blocks_for(batch),
                recipe.seed, execution.prices(), execution.standard_errors());
        });
}
