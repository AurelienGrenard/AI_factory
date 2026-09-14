// Generated Cartesian-product rough_sabr cliquet paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/cliquet_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "product/cliquet/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/cliquet/cliquets_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/cliquets/rough_sabr_01__cliquets_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/cliquets/rough_sabr_01__cliquets_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/cliquets/rough_sabr_01__cliquets_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/cliquets/rough_sabr_01__cliquets_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::RegularHybridSchedule, product::CliquetPathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "cliquet", ""},
        11668828794885505024ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, product::load_cliquets,
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_cliquet_price_delta_cuda(arguments...);
        });
}
