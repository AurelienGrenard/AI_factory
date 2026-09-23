// Generated rough_sabr geometric_asian_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/geometric_asian_option_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "product/geometric_asian_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/geometric_asian_calls/rough_sabr_01__geometric_asian_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/geometric_asian_calls/rough_sabr_01__geometric_asian_calls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/geometric_asian_calls/rough_sabr_01__geometric_asian_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/geometric_asian_calls/rough_sabr_01__geometric_asian_calls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::GeometricAsianOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "geometric_asian_option", ""},
        11668828850720079872ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_geometric_asian_options,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_geometric_asian_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
