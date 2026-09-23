// Generated rough_stein_stein athena_autocall paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/athena_autocall_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "product/athena_autocall/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/athena_autocalls/rough_stein_stein_01__athena_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/athena_autocalls/rough_stein_stein_01__athena_autocalls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/athena_autocalls/rough_stein_stein_01__athena_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/athena_autocalls/rough_stein_stein_01__athena_autocalls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::AthenaAutocallPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "athena_autocall", ""},
        11668828923734523904ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_athena_autocalls,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_athena_autocall_price_delta_cuda(arguments...);
        });
}
