// Generated log_modulated_rough_bergomi asset_or_nothing_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/asset_or_nothing_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/asset_or_nothing_option/dataset.hpp"
#include "product/asset_or_nothing_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/asset_or_nothing_puts/log_modulated_rough_bergomi_01__asset_or_nothing_puts_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/asset_or_nothing_puts/log_modulated_rough_bergomi_01__asset_or_nothing_puts_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/asset_or_nothing_puts/log_modulated_rough_bergomi_01__asset_or_nothing_puts_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/asset_or_nothing_puts/log_modulated_rough_bergomi_01__asset_or_nothing_puts_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::AssetOrNothingOptionPathPolicy<OptionSide::put>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "asset_or_nothing_option", ""},
        11668828253719625728ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_asset_or_nothing_options,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_asset_or_nothing_option_price_delta_cuda<OptionSide::put>(arguments...);
        });
}
