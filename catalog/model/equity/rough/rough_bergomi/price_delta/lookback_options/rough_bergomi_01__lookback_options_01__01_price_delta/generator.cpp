// Generated rough_bergomi lookback_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/lookback_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/lookback_option/dataset.hpp"
#include "product/lookback_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/lookback_option/lookback_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/lookback_options/rough_bergomi_01__lookback_options_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/lookback_options/rough_bergomi_01__lookback_options_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/lookback_options/rough_bergomi_01__lookback_options_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/lookback_options/rough_bergomi_01__lookback_options_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::LookbackOptionPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "lookback_option", ""},
        11668828593022042112ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_lookback_options,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_lookback_option_price_delta_cuda(arguments...);
        });
}
