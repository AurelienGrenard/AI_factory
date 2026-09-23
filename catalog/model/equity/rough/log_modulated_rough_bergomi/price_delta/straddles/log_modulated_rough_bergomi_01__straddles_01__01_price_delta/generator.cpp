// Generated log_modulated_rough_bergomi straddle paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/straddle_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "product/straddle/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/straddle/straddles_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/straddles/log_modulated_rough_bergomi_01__straddles_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/straddles/log_modulated_rough_bergomi_01__straddles_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/straddles/log_modulated_rough_bergomi_01__straddles_01__01_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/straddles/log_modulated_rough_bergomi_01__straddles_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_volterra_price_delta_dataset<volterra::TerminalHybridSchedule, product::StraddlePathPolicy>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "straddle", ""},
        11668828343913938944ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_straddles,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_straddle_price_delta_cuda(arguments...);
        });
}
