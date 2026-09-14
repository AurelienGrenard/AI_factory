// Generated Cartesian-product rough_stein_stein geometric_asian_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/geometric_asian_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "product/geometric_asian_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/geometric_asian_calls/rough_stein_stein_01__geometric_asian_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/geometric_asian_calls/rough_stein_stein_01__geometric_asian_calls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/geometric_asian_calls/rough_stein_stein_01__geometric_asian_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/geometric_asian_calls/rough_stein_stein_01__geometric_asian_calls_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::GeometricAsianOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "geometric_asian_option", ""},
        11668828983864066048ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_geometric_asian_options,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_geometric_asian_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
