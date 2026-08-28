// Generated {model_display} {variant_comment} price-dataset recipe.
#include "model/equity/markovian/{model}/product/{product}.cuh"
#include "model/equity/markovian/{model}/dataset.hpp"
#include "product/{product}/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {{
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::{model};
    namespace pricing = offline::pricing;

{time_constants}
    const pricing::EquityPriceRecipe recipe{{
        "{model_dataset_path}",
        "{product_dataset_path}",
        "{price_dataset_path}",
        "{catalog_path}",
        "{url}",
        "{numerical_method}",
        PriceConstruction::Aligned,
    }};
    const pricing::BatchedMonteCarloProfile profile{{
        {monte_carlo_paths}U,
        4'096U,
        4'096U,
        {threads_per_block},
        {seed}ULL,
        "{delta_t_description}",
        {execution_metadata},
    }};

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        {product_loader_expression},
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::BatchedMonteCarloProfile& execution_profile) {{
            return pricing::execute_batched_monte_carlo(
                models,
                products,
                construction,
                execution_profile,
                [&](const auto* device_models, std::size_t model_count,
                    const auto* device_products, std::size_t product_count,
                    const pricing::BatchedLaunchContext& context,
                    float* device_prices,
                    float* device_standard_errors) {{
                    model_binding::launch_{model}_{product}_cuda{side_template}(
                        device_models,
                        model_count,
                        device_products,
                        product_count,
                        context.construction,
                        context.result_count,
                        context.result_offset,
                        context.launch_result_count,
                        context.paths_per_price,
{time_arguments}                        execution_profile.threads_per_block,
                        context.block_count,
                        context.seed,
                        device_prices,
                        device_standard_errors
                    );
                }}
            );
        }}
    );
}}
