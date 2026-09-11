// Generated rough_bergomi double_knock_out_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/rough_bergomi/product/double_knock_out_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "product/double_knock_out_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "datasets/product/double_knock_out_option/double_knock_out_options_01.json", "datasets/model/equity/rough/rough_bergomi/price_delta/double_knock_out_calls/rough_bergomi_01__double_knock_out_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/price_delta/double_knock_out_calls/rough_bergomi_01__double_knock_out_calls_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/price_delta/double_knock_out_calls/rough_bergomi_01__double_knock_out_calls_01__01_price_delta.json", "catalog/model/equity/rough/rough_bergomi/prices/double_knock_out_calls/rough_bergomi_01__double_knock_out_calls_01__01/generator.cpp", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::DoubleKnockOutOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "rough_bergomi", "double_knock_out_option", ""},
        11668828541482434560ULL, "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)", model::equity::rough_bergomi::load_models, product::load_double_knock_out_options,
        [](auto... arguments) {
            model::equity::rough_bergomi::launch_rough_bergomi_double_knock_out_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
