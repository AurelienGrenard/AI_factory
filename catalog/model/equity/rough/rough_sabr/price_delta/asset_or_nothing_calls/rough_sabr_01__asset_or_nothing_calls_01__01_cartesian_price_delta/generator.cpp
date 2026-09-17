// Generated Cartesian-product rough_sabr asset_or_nothing_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/asset_or_nothing_option_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "product/asset_or_nothing_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/asset_or_nothing_calls/rough_sabr_01__asset_or_nothing_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/asset_or_nothing_calls/rough_sabr_01__asset_or_nothing_calls_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/asset_or_nothing_calls/rough_sabr_01__asset_or_nothing_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/asset_or_nothing_calls/rough_sabr_01__asset_or_nothing_calls_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::AssetOrNothingOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "asset_or_nothing_option", ""},
        11668828782000603136ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_asset_or_nothing_options,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_asset_or_nothing_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
