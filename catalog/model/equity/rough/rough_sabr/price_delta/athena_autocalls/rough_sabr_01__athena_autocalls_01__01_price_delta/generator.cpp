// Generated rough_sabr athena_autocall paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/athena_autocall_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "product/athena_autocall/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/athena_autocalls/rough_sabr_01__athena_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/athena_autocalls/rough_sabr_01__athena_autocalls_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/athena_autocalls/rough_sabr_01__athena_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/athena_autocalls/rough_sabr_01__athena_autocalls_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::AthenaAutocallPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "athena_autocall", ""},
        11668828790590537728ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_athena_autocalls,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_athena_autocall_price_delta_cuda(arguments...);
        });
}
