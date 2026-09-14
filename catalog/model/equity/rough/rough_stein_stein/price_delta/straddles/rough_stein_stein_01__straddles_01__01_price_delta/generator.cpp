// Generated rough_stein_stein straddle paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/straddle_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "product/straddle/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/straddles/rough_stein_stein_01__straddles_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/straddles/rough_stein_stein_01__straddles_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/straddles/rough_stein_stein_01__straddles_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/straddles/rough_stein_stein_01__straddles_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::StraddlePathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "straddle", ""},
        11668829009633869824ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_straddles,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_straddle_price_delta_cuda(arguments...);
        });
}
