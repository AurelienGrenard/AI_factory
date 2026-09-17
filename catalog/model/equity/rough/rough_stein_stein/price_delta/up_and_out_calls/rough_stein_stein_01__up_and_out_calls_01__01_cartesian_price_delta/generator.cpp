// Generated Cartesian-product rough_stein_stein up_and_out_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/up_and_out_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/up_and_out_option/dataset.hpp"
#include "product/up_and_out_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/up_and_out_option/up_and_out_options_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/up_and_out_calls/rough_stein_stein_01__up_and_out_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/up_and_out_calls/rough_stein_stein_01__up_and_out_calls_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/up_and_out_calls/rough_stein_stein_01__up_and_out_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/up_and_out_calls/rough_stein_stein_01__up_and_out_calls_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::UpAndOutOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "up_and_out_option", ""},
        11668829018223804416ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_up_and_out_options,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_up_and_out_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
