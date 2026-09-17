// Generated Cartesian-product rough_bergomi european_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "product/european_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/european_calls/rough_bergomi_01__european_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/european_calls/rough_bergomi_01__european_calls_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/european_calls/rough_bergomi_01__european_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/european_calls/rough_bergomi_01__european_calls_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::EuropeanOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "european_option", ""},
        11668828558662303744ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_european_options,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_european_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
