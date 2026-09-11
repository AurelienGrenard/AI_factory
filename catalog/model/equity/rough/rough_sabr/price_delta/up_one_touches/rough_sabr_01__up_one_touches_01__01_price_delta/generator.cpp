// Generated rough_sabr up_one_touch paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/up_one_touch_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "product/up_one_touch/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/up_one_touches/rough_sabr_01__up_one_touches_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/up_one_touches/rough_sabr_01__up_one_touches_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/up_one_touches/rough_sabr_01__up_one_touches_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/up_one_touches/rough_sabr_01__up_one_touches_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::UpOneTouchPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "up_one_touch", ""},
        11668828893669752832ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_up_one_touches,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_up_one_touch_price_delta_cuda(arguments...);
        });
}
