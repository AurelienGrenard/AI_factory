// Generated log_modulated_rough_bergomi up_and_out_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/up_and_out_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/up_and_out_option/dataset.hpp"
#include "product/up_and_out_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/up_and_out_option/up_and_out_options_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/up_and_out_calls/log_modulated_rough_bergomi_01__up_and_out_calls_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/up_and_out_calls/log_modulated_rough_bergomi_01__up_and_out_calls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/up_and_out_calls/log_modulated_rough_bergomi_01__up_and_out_calls_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/up_and_out_calls/log_modulated_rough_bergomi_01__up_and_out_calls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::UpAndOutOptionPathPolicy<OptionSide::call>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "up_and_out_option", ""},
        11668828352503873536ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_up_and_out_options,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_up_and_out_option_price_delta_cuda<OptionSide::call>(arguments...);
        });
}
