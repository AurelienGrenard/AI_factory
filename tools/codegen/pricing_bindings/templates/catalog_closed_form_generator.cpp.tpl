// Generated Black-Scholes {variant_comment} analytical price recipe.
#include "model/equity/markovian/black_scholes/{product}.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/{product}/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>

int main() {{
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::black_scholes;
    namespace pricing = offline::pricing;

    const pricing::EquityPriceRecipe recipe{{
        "{model_dataset_path}",
        "{product_dataset_path}",
        "{price_dataset_path}",
        "{catalog_path}",
        "{url}",
        "Black-Scholes closed-form {variant_comment}",
        PriceConstruction::Aligned,
    }};
    const pricing::AnalyticalProfile profile{{{analytical_profile_values}}};

    return pricing::generate_analytical_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        {product_loader_expression},
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::AnalyticalProfile& execution_profile) {{
            return pricing::execute_analytical(
                models,
                products,
                construction,
                execution_profile,
                [&](const auto* device_models, std::size_t model_count,
                    const auto* device_products, std::size_t product_count,
                    const pricing::AnalyticalLaunchContext& context,
                    float* device_prices) {{
                    model_binding::launch_black_scholes_{product}_cuda{side_template}(
                        device_models,
                        model_count,
                        device_products,
                        product_count,
                        context.construction,
                        context.result_count,
                        context.result_offset,
                        context.launch_result_count,
{analytical_time_arguments}
                        context.threads_per_block,
                        context.block_count,
                        device_prices
                    );
                }}
            );
        }}
    );
}}
