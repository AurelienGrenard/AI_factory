// Generated rough_stein_stein double_knock_out_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_stein_stein/product/double_knock_out_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "product/double_knock_out_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json", "datasets/product/double_knock_out_option/double_knock_out_options_01.json", "datasets/model/equity/rough/rough_stein_stein/price_delta/double_knock_out_calls/rough_stein_stein_01__double_knock_out_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/price_delta/double_knock_out_calls/rough_stein_stein_01__double_knock_out_calls_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/price_delta/double_knock_out_calls/rough_stein_stein_01__double_knock_out_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_stein_stein/prices/double_knock_out_calls/rough_stein_stein_01__double_knock_out_calls_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::DoubleKnockOutOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_stein_stein", "double_knock_out_option", ""},
        11668828940914393088ULL, "fractional-resolvent hybrid FFT", model::equity::rough_stein_stein::load_models, product::load_double_knock_out_options,
        [](auto... arguments) {
            model::equity::rough_stein_stein::launch_rough_stein_stein_double_knock_out_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
