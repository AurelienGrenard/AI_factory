// Generated Cartesian-product rough_sabr european_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "product/european_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/european_puts/rough_sabr_01__european_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/european_puts/rough_sabr_01__european_puts_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/european_puts/rough_sabr_01__european_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/european_puts/rough_sabr_01__european_puts_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::EuropeanOptionPathPolicy<OptionSide::put>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "european_option", ""},
        11668828829245243392ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_european_options,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_european_option_price_delta_cuda<OptionSide::put>(arguments...);
        });
}
