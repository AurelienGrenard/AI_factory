// Generated log_modulated_rough_bergomi digital_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/digital_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "product/digital_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/digital_option/digital_options_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/digital_calls/log_modulated_rough_bergomi_01__digital_calls_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/digital_calls/log_modulated_rough_bergomi_01__digital_calls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/digital_calls/log_modulated_rough_bergomi_01__digital_calls_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/digital_calls/log_modulated_rough_bergomi_01__digital_calls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::DigitalOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "digital_option", ""},
        11668828266604527616ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_digital_options,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_digital_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
