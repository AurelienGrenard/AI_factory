// Generated rough_sabr straddle paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/straddle_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "product/straddle/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/straddles/rough_sabr_01__straddles_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/straddles/rough_sabr_01__straddles_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/straddles/rough_sabr_01__straddles_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/straddles/rough_sabr_01__straddles_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::StraddlePathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "straddle", ""},
        11668828876489883648ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_straddles,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_straddle_price_delta_cuda(arguments...);
        });
}
