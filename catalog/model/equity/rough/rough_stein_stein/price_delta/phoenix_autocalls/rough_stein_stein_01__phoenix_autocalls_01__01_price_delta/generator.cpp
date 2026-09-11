// Generated rough_stein_stein phoenix_autocall paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/phoenix_autocall_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "product/phoenix_autocall/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/phoenix_autocalls/rough_stein_stein_01__phoenix_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/phoenix_autocalls/rough_stein_stein_01__phoenix_autocalls_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/phoenix_autocalls/rough_stein_stein_01__phoenix_autocalls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/phoenix_autocalls/rough_stein_stein_01__phoenix_autocalls_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::PhoenixAutocallPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "phoenix_autocall", ""},
        11668828996748967936ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_phoenix_autocalls,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_phoenix_autocall_price_delta_cuda(arguments...);
        });
}
