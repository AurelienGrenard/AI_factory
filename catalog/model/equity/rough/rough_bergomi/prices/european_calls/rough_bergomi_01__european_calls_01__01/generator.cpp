// Generated Rough-Bergomi european-calls Volterra-FFT price recipe.
#include "model/equity/rough/rough_bergomi/product/european_option.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "model/equity/rough/rough_bergomi/volterra_fft_workspace.cuh"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::rough_bergomi;
    namespace pricing = offline::pricing;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json",
        "datasets/product/european_option/european_options_01.json",
        "datasets/model/equity/rough/rough_bergomi/prices/european_calls/rough_bergomi_01__european_calls_01__01.json",
        "catalog/model/equity/rough/rough_bergomi/prices/european_calls/rough_bergomi_01__european_calls_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_bergomi/prices/european_calls/rough_bergomi_01__european_calls_01__01.json",
        "Bennedsen-Lunde-Pakkanen hybrid FFT (kappa=1)",
        PriceConstruction::Aligned,
    };
    const pricing::VolterraMonteCarloProfile profile{
        1'048'576U,
        1.0f / 252.0f,
        1.0f / 504.0f,
        ::ai_factory::workbench::offline::cuda_tuning::kVolterraPathChunkSize,
        11668828558662303744ULL,
        "1 / 504",
    };

    return pricing::generate_volterra_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        product::load_european_options,
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
                    model_binding::launch_rough_bergomi_european_option_cuda<OptionSide::call>(
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
