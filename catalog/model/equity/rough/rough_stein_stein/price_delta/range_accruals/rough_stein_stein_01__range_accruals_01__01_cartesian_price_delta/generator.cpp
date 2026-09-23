// Generated Cartesian-product rough_stein_stein range_accrual paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/range_accrual_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "product/range_accrual/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/range_accruals/rough_stein_stein_01__range_accruals_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/range_accruals/rough_stein_stein_01__range_accruals_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/range_accruals/rough_stein_stein_01__range_accruals_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/range_accruals/rough_stein_stein_01__range_accruals_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::RangeAccrualPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "range_accrual", ""},
        11668829005338902528ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_range_accruals,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_range_accrual_price_delta_cuda(arguments...);
        });
}
