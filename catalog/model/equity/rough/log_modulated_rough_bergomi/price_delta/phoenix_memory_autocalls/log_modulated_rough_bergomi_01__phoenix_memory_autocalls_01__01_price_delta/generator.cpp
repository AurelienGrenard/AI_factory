// Generated log_modulated_rough_bergomi phoenix_memory_autocall paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/phoenix_memory_autocall_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/phoenix_memory_autocalls/log_modulated_rough_bergomi_01__phoenix_memory_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/phoenix_memory_autocalls/log_modulated_rough_bergomi_01__phoenix_memory_autocalls_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/phoenix_memory_autocalls/log_modulated_rough_bergomi_01__phoenix_memory_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/phoenix_memory_autocalls/log_modulated_rough_bergomi_01__phoenix_memory_autocalls_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::PhoenixMemoryAutocallPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "phoenix_memory_autocall", ""},
        11668828335324004352ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_phoenix_memory_autocalls,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_phoenix_memory_autocall_price_delta_cuda(arguments...);
        });
}
