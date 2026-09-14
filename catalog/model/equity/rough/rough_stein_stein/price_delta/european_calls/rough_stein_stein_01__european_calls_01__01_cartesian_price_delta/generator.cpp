// Generated Cartesian-product rough_stein_stein european_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "product/european_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/european_option/european_options_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/european_calls/rough_stein_stein_01__european_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/european_calls/rough_stein_stein_01__european_calls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/european_calls/rough_stein_stein_01__european_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/european_calls/rough_stein_stein_01__european_calls_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::EuropeanOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "european_option", ""},
        11668828958094262272ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_european_options,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_european_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
