// Generated Cartesian-product rough_bergomi up_and_in_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/up_and_in_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/up_and_in_option/dataset.hpp"
#include "product/up_and_in_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/up_and_in_option/up_and_in_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/up_and_in_calls/rough_bergomi_01__up_and_in_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/up_and_in_calls/rough_bergomi_01__up_and_in_calls_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/up_and_in_calls/rough_bergomi_01__up_and_in_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/up_and_in_calls/rough_bergomi_01__up_and_in_calls_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::UpAndInOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "up_and_in_option", ""},
        11668828614496878592ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_up_and_in_options,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_up_and_in_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
