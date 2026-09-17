// Generated Cartesian-product rough_stein_stein asian_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/asian_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/asian_option/dataset.hpp"
#include "product/asian_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/asian_option/asian_options_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/asian_calls/rough_stein_stein_01__asian_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/asian_calls/rough_stein_stein_01__asian_calls_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/asian_calls/rough_stein_stein_01__asian_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/asian_calls/rough_stein_stein_01__asian_calls_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::AsianOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "asian_option", ""},
        11668828906554654720ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_asian_options,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_asian_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
