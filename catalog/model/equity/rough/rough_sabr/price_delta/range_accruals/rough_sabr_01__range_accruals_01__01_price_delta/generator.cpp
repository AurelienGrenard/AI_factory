// Generated rough_sabr range_accrual paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/range_accrual_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "product/range_accrual/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/range_accruals/rough_sabr_01__range_accruals_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/range_accruals/rough_sabr_01__range_accruals_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/range_accruals/rough_sabr_01__range_accruals_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/range_accruals/rough_sabr_01__range_accruals_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::RangeAccrualPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "range_accrual", ""},
        11668828872194916352ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_range_accruals,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_range_accrual_price_delta_cuda(arguments...);
        });
}
