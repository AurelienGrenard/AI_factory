// Generated rough_sabr phoenix_autocall paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/phoenix_autocall_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "product/phoenix_autocall/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/phoenix_autocalls/rough_sabr_01__phoenix_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/phoenix_autocalls/rough_sabr_01__phoenix_autocalls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/phoenix_autocalls/rough_sabr_01__phoenix_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/phoenix_autocalls/rough_sabr_01__phoenix_autocalls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::PhoenixAutocallPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "phoenix_autocall", ""},
        11668828863604981760ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_phoenix_autocalls,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_phoenix_autocall_price_delta_cuda(arguments...);
        });
}
