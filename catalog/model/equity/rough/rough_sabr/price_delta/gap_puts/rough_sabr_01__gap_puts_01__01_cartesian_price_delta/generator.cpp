// Generated Cartesian-product rough_sabr gap_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_sabr/product/gap_option_price_delta.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "product/gap_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json", "datasets/product/gap_option/gap_put_options_01.json", "datasets/model/equity/rough/rough_sabr/price_delta/gap_puts/rough_sabr_01__gap_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/price_delta/gap_puts/rough_sabr_01__gap_puts_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_sabr/price_delta/gap_puts/rough_sabr_01__gap_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/rough_sabr/prices/gap_puts/rough_sabr_01__gap_puts_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::GapOptionPathPolicy<OptionSide::put>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_sabr", "gap_option", ""},
        11668828846425112576ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot", model::equity::rough_sabr::load_models, [](const auto& path) { return product::load_gap_options(path, OptionSide::put); },
        [](auto... arguments) {
            model::equity::rough_sabr::launch_rough_sabr_gap_option_price_delta_cuda<OptionSide::put>(arguments...);
        });
}
