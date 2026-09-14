// Generated Cartesian-product rough_bergomi range_accrual paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/range_accrual_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "product/range_accrual/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/range_accruals/rough_bergomi_01__range_accruals_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/range_accruals/rough_bergomi_01__range_accruals_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/range_accruals/rough_bergomi_01__range_accruals_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/range_accruals/rough_bergomi_01__range_accruals_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::RangeAccrualPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "range_accrual", ""},
        11668828605906944000ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_range_accruals,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_range_accrual_price_delta_cuda(arguments...);
        });
}
