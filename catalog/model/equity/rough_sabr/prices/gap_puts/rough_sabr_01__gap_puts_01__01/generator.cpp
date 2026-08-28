// Generated Rough-SABR gap-puts Volterra-FFT price recipe.
#include "model/equity/rough/rough_sabr/product/gap_option.cuh"
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "model/equity/rough/rough_sabr/volterra_fft_workspace.cuh"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::rough_sabr;
    namespace pricing = offline::pricing;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/rough_sabr/parameters/rough_sabr_01.json",
        "datasets/product/equity/gap_options/gap_put_options_01.json",
        "datasets/model/equity/rough_sabr/prices/gap_puts/rough_sabr_01__gap_puts_01__01.json",
        "catalog/model/equity/rough_sabr/prices/gap_puts/rough_sabr_01__gap_puts_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough_sabr/prices/gap_puts/rough_sabr_01__gap_puts_01__01.json",
        "Bennedsen-Lunde-Pakkanen hybrid FFT with Lamperti spot",
        PriceConstruction::Aligned,
    };
    const pricing::VolterraMonteCarloProfile profile{
        1'048'576U,
        1.0f / 252.0f,
        1.0f / 504.0f,
        ::ai_factory::workbench::offline::cuda_tuning::kVolterraPathChunkSize,
        910000001ULL,
        "1 / 504",
    };

    return pricing::generate_volterra_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        [](const auto& path) { return product::load_gap_options(path, OptionSide::put); },
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::VolterraMonteCarloProfile& execution_profile) {
            return pricing::execute_volterra_monte_carlo(
                models,
                products,
                construction,
                execution_profile,
                model_binding::plan_pricing_workspace,
                [&](const auto* device_models, std::size_t model_count,
                    const auto* device_products, std::size_t product_count,
                    const pricing::VolterraLaunchContext& context,
                    void* device_workspace,
                    std::size_t workspace_bytes,
                    float* device_prices,
                    float* device_standard_errors) {
                    model_binding::launch_rough_sabr_gap_option_cuda<OptionSide::put>(
                        device_models,
                        model_count,
                        device_products,
                        product_count,
                        context.construction,
                        context.result_count,
                        context.result_index,
                        context.paths_per_price,
                        context.day_fraction,
                        context.target_dt,
                        context.step_count,
                        context.path_chunk_size,
                        device_workspace,
                        workspace_bytes,
                        context.seed,
                        device_prices,
                        device_standard_errors
                    );
                }
            );
        }
    );
}
