// Generated rough_bergomi forward_start_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/forward_start_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "product/forward_start_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/forward_start_puts/rough_bergomi_01__forward_start_puts_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/forward_start_puts/rough_bergomi_01__forward_start_puts_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/forward_start_puts/rough_bergomi_01__forward_start_puts_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/forward_start_puts/rough_bergomi_01__forward_start_puts_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::CalendarHybridSchedule<2U>, product::ForwardStartOptionPathPolicy<OptionSide::put>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "forward_start_option", ""},
        11668828571547205632ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_forward_start_options,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_forward_start_option_price_delta_cuda<OptionSide::put>(arguments...);
        });
}
