// Generated rough_bergomi asset_or_nothing_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/asset_or_nothing_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "product/asset_or_nothing_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/asset_or_nothing_calls/rough_bergomi_01__asset_or_nothing_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/asset_or_nothing_calls/rough_bergomi_01__asset_or_nothing_calls_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/asset_or_nothing_calls/rough_bergomi_01__asset_or_nothing_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/asset_or_nothing_calls/rough_bergomi_01__asset_or_nothing_calls_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::AssetOrNothingOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "asset_or_nothing_option", ""},
        11668828515712630784ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_asset_or_nothing_options,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_asset_or_nothing_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
