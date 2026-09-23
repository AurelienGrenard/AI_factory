// Generated rough_stein_stein phoenix_memory_autocall paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/phoenix_memory_autocall_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/phoenix_memory_autocalls/rough_stein_stein_01__phoenix_memory_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/phoenix_memory_autocalls/rough_stein_stein_01__phoenix_memory_autocalls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/phoenix_memory_autocalls/rough_stein_stein_01__phoenix_memory_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/phoenix_memory_autocalls/rough_stein_stein_01__phoenix_memory_autocalls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::PhoenixMemoryAutocallPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "phoenix_memory_autocall", ""},
        11668829001043935232ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_phoenix_memory_autocalls,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_phoenix_memory_autocall_price_delta_cuda(arguments...);
        });
}
