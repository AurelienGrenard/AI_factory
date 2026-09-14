// Generated log_modulated_rough_bergomi range_accrual paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/range_accrual_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "product/range_accrual/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/range_accruals/log_modulated_rough_bergomi_01__range_accruals_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/range_accruals/log_modulated_rough_bergomi_01__range_accruals_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/range_accruals/log_modulated_rough_bergomi_01__range_accruals_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/range_accruals/log_modulated_rough_bergomi_01__range_accruals_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::RangeAccrualPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "range_accrual", ""},
        11668828339618971648ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_range_accruals,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_range_accrual_price_delta_cuda(arguments...);
        });
}
