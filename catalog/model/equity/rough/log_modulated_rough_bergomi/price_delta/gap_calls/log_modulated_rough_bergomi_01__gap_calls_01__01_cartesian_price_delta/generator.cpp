// Generated Cartesian-product log_modulated_rough_bergomi gap_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/gap_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "product/gap_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/gap_option/gap_call_options_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/gap_calls/log_modulated_rough_bergomi_01__gap_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/gap_calls/log_modulated_rough_bergomi_01__gap_calls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/gap_calls/log_modulated_rough_bergomi_01__gap_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/gap_calls/log_modulated_rough_bergomi_01__gap_calls_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::GapOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "gap_option", ""},
        11668828309554200576ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, [](const auto& path) { return product::load_gap_options(path, OptionSide::call); },
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_gap_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
