// Generated Cartesian-product log_modulated_rough_bergomi athena_autocall paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/athena_autocall_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "product/athena_autocall/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/athena_autocalls/log_modulated_rough_bergomi_01__athena_autocalls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/athena_autocalls/log_modulated_rough_bergomi_01__athena_autocalls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/athena_autocalls/log_modulated_rough_bergomi_01__athena_autocalls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/athena_autocalls/log_modulated_rough_bergomi_01__athena_autocalls_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::AthenaAutocallPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "athena_autocall", ""},
        11668828258014593024ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_athena_autocalls,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_athena_autocall_price_delta_cuda(arguments...);
        });
}
