// Generated Cartesian-product log_modulated_rough_bergomi geometric_asian_option paired recipe with one shared FFT workspace.
#include "model/equity/rough/log_modulated_rough_bergomi/product/geometric_asian_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "product/geometric_asian_option/pricing_policy.cuh"
#include "tools/pricing/volterra_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/rough/log_modulated_rough_bergomi/price_delta/geometric_asian_puts/log_modulated_rough_bergomi_01__geometric_asian_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/price_delta/geometric_asian_puts/log_modulated_rough_bergomi_01__geometric_asian_puts_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/price_delta/geometric_asian_puts/log_modulated_rough_bergomi_01__geometric_asian_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/log_modulated_rough_bergomi/prices/geometric_asian_puts/log_modulated_rough_bergomi_01__geometric_asian_puts_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_volterra_price_delta_dataset<volterra::DenseHybridSchedule, product::GeometricAsianOptionPathPolicy<OptionSide::put>>(
        recipe, {offline::cuda_tuning::PricingFamily::rough_fft, "log_modulated_rough_bergomi", "geometric_asian_option", ""},
        11668828322439102464ULL, "log-modulated hybrid FFT (kappa=1)", model::equity::log_modulated_rough_bergomi::load_models, product::load_geometric_asian_options,
        [](auto... arguments) {
            model::equity::log_modulated_rough_bergomi::launch_log_modulated_rough_bergomi_geometric_asian_option_price_delta_cuda<OptionSide::put>(arguments...);
        });
}
