// Generated log_modulated_rough_bergomi up_one_touch paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/up_one_touch_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "product/up_one_touch/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/up_one_touches/log_modulated_rough_bergomi_01__up_one_touches_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/up_one_touches/log_modulated_rough_bergomi_01__up_one_touches_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/up_one_touches/log_modulated_rough_bergomi_01__up_one_touches_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/up_one_touches/log_modulated_rough_bergomi_01__up_one_touches_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::UpOneTouchPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "up_one_touch", ""},
        11668828361093808128ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_up_one_touches,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_up_one_touch_price_delta_cuda(arguments...);
        });
}
