// Generated rough_stein_stein forward_start_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/forward_start_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "product/forward_start_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/forward_start_option/forward_start_options_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/forward_start_calls/rough_stein_stein_01__forward_start_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/forward_start_calls/rough_stein_stein_01__forward_start_calls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/forward_start_calls/rough_stein_stein_01__forward_start_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/forward_start_calls/rough_stein_stein_01__forward_start_calls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::CalendarHybridSchedule<2U>, product::ForwardStartOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "forward_start_option", ""},
        11668828966684196864ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_forward_start_options,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_forward_start_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
