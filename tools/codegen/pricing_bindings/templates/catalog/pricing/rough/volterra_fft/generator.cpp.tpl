// Generated {model_display} {variant_comment} Volterra-FFT price recipe.
#include "model/equity/rough/{model}/product/{product}.cuh"
#include "model/equity/rough/{model}/dataset.hpp"
#include "model/equity/rough/{model}/volterra_fft_workspace.cuh"
#include "product/{product}/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>

int main() {{
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::{model};
    namespace pricing = offline::pricing;

    const pricing::EquityPriceRecipe recipe{{
        "{model_dataset_path}",
        "{product_dataset_path}",
        "{price_dataset_path}",
        "{catalog_path}",
        "{url}",
        "{numerical_method}",
        PriceConstruction::Aligned,
    }};
    const pricing::VolterraMonteCarloProfile profile{{
        {monte_carlo_paths}U,
        1.0f / 252.0f,
        1.0f / 504.0f,
        ::ai_factory::workbench::offline::cuda_tuning::kVolterraPathChunkSize,
        {seed}ULL,
        "1 / 504",
    }};

    return pricing::generate_volterra_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        {product_loader_expression},
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::VolterraMonteCarloProfile& execution_profile) {{
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
                    float* device_standard_errors) {{
                    model_binding::launch_{model}_{product}_cuda{side_template}(
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
                }}
            );
        }}
    );
}}
