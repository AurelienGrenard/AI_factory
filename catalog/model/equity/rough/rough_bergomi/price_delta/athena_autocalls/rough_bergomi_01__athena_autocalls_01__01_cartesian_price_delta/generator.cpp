// Generated Cartesian-product rough_bergomi athena_autocall paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/athena_autocall_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "product/athena_autocall/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/athena_autocalls/rough_bergomi_01__athena_autocalls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/athena_autocalls/rough_bergomi_01__athena_autocalls_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/athena_autocalls/rough_bergomi_01__athena_autocalls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/athena_autocalls/rough_bergomi_01__athena_autocalls_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::AthenaAutocallPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "athena_autocall", ""},
        11668828524302565376ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_athena_autocalls,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_athena_autocall_price_delta_cuda(arguments...);
        });
}
