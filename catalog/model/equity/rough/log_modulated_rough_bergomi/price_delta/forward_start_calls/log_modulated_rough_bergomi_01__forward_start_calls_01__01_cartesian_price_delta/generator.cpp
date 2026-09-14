// Generated Cartesian-product log_modulated_rough_bergomi forward_start_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/forward_start_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "product/forward_start_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/forward_start_calls/log_modulated_rough_bergomi_01__forward_start_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/forward_start_calls/log_modulated_rough_bergomi_01__forward_start_calls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/forward_start_calls/log_modulated_rough_bergomi_01__forward_start_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/forward_start_calls/log_modulated_rough_bergomi_01__forward_start_calls_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::CalendarHybridSchedule<2U>, product::ForwardStartOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "forward_start_option", ""},
        11668828300964265984ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_forward_start_options,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_forward_start_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
