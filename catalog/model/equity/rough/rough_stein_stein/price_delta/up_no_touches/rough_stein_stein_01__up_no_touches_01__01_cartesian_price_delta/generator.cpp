// Generated Cartesian-product rough_stein_stein up_no_touch paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/up_no_touch_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/up_no_touch/dataset.hpp"
#include "product/up_no_touch/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/up_no_touch/up_no_touches_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/up_no_touches/rough_stein_stein_01__up_no_touches_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/up_no_touches/rough_stein_stein_01__up_no_touches_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/up_no_touches/rough_stein_stein_01__up_no_touches_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/up_no_touches/rough_stein_stein_01__up_no_touches_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::UpNoTouchPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "up_no_touch", ""},
        11668829022518771712ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_up_no_touches,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_up_no_touch_price_delta_cuda(arguments...);
        });
}
