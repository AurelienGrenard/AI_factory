// Generated log_modulated_rough_bergomi european_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/european_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "product/european_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/european_calls/log_modulated_rough_bergomi_01__european_calls_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/european_calls/log_modulated_rough_bergomi_01__european_calls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/european_calls/log_modulated_rough_bergomi_01__european_calls_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/european_calls/log_modulated_rough_bergomi_01__european_calls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::EuropeanOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "european_option", ""},
        11668828292374331392ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_european_options,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_european_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
