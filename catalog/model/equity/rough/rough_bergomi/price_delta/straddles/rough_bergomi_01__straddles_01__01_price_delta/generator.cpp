// Generated rough_bergomi straddle paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/straddle_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "product/straddle/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/straddles/rough_bergomi_01__straddles_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/straddles/rough_bergomi_01__straddles_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/straddles/rough_bergomi_01__straddles_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/straddles/rough_bergomi_01__straddles_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::StraddlePathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "straddle", ""},
        11668828610201911296ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_straddles,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_straddle_price_delta_cuda(arguments...);
        });
}
