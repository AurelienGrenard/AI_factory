// Generated Cartesian-product rough_bergomi asian_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/asian_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/asian_option/dataset.hpp"
#include "product/asian_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/asian_option/asian_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/asian_puts/rough_bergomi_01__asian_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/asian_puts/rough_bergomi_01__asian_puts_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/asian_puts/rough_bergomi_01__asian_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/asian_puts/rough_bergomi_01__asian_puts_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::AsianOptionPathPolicy<OptionSide::put>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "asian_option", ""},
        11668828511417663488ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_asian_options,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_asian_option_price_delta_cuda<OptionSide::put>(arguments...);
        });
}
