// Generated Cartesian-product log_modulated_rough_bergomi cliquet paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/cliquet_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "product/cliquet/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/cliquets/log_modulated_rough_bergomi_01__cliquets_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/cliquets/log_modulated_rough_bergomi_01__cliquets_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/cliquets/log_modulated_rough_bergomi_01__cliquets_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/cliquets/log_modulated_rough_bergomi_01__cliquets_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::CliquetPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "cliquet", ""},
        11668828262309560320ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_cliquets,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_cliquet_price_delta_cuda(arguments...);
        });
}
