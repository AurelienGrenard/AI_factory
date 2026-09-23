// Generated rough_stein_stein asset_or_nothing_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/asset_or_nothing_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "product/asset_or_nothing_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/asset_or_nothing_puts/rough_stein_stein_01__asset_or_nothing_puts_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/asset_or_nothing_puts/rough_stein_stein_01__asset_or_nothing_puts_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/asset_or_nothing_puts/rough_stein_stein_01__asset_or_nothing_puts_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/asset_or_nothing_puts/rough_stein_stein_01__asset_or_nothing_puts_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::AssetOrNothingOptionPathPolicy<OptionSide::put>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "asset_or_nothing_option", ""},
        11668828919439556608ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_asset_or_nothing_options,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_asset_or_nothing_option_price_delta_cuda<OptionSide::put>(arguments...);
        });
}
