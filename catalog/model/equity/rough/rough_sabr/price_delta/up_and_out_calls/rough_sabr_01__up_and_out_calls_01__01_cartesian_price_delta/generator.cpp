// Generated Cartesian-product rough_sabr up_and_out_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/up_and_out_option_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/up_and_out_option/dataset.hpp"
#include "product/up_and_out_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/up_and_out_option/up_and_out_options_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/up_and_out_calls/rough_sabr_01__up_and_out_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/up_and_out_calls/rough_sabr_01__up_and_out_calls_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/up_and_out_calls/rough_sabr_01__up_and_out_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/up_and_out_calls/rough_sabr_01__up_and_out_calls_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::UpAndOutOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "up_and_out_option", ""},
        11668828885079818240ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_up_and_out_options,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_up_and_out_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
