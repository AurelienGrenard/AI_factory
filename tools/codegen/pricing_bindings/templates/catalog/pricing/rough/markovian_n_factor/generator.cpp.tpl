// Generated {model_display} {variant_comment} N-factor price recipe.
#include "model/equity/rough/{model}/product/{product}.cuh"
#include "model/equity/rough/{model}/dataset.hpp"
#include "model/equity/rough/{model}/markovian_n_factor_preparation.hpp"
#include "product/{product}/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>

int main() {{
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::{model};
    namespace pricing = offline::pricing;

    constexpr std::size_t factor_count = 7U;
    constexpr float day_fraction = 1.0f / 252.0f;
    constexpr float dt = 1.0f / 504.0f;
    constexpr std::uint32_t simulation_steps_per_day = 2U;
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
        "1 / 504",
        nlohmann::ordered_json{{
            {{"simulation_steps_per_day", simulation_steps_per_day}},
            {{"factor_count", factor_count}},
        }},
    }};

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        {product_loader_expression},
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::BatchedMonteCarloProfile& execution_profile) {{
            float approximation_horizon = day_fraction;
            for (const auto& product : products) {{
                approximation_horizon = std::max(
                    approximation_horizon,
                    static_cast<float>(product.maturity_days) * day_fraction
                );
            }}
            const auto prepared = model_binding::prepare_dynamics<factor_count>(
                models,
                approximation_horizon,
                dt
            );
            return pricing::execute_prepared_batched_monte_carlo(
                models,
                prepared,
                products,
                construction,
                execution_profile,
                [&](const auto* device_models, std::size_t model_count,
                    const auto* device_prepared,
                    std::size_t prepared_count,
                    const auto* device_products, std::size_t product_count,
                    const pricing::BatchedLaunchContext& context,
                    float* device_prices,
                    float* device_standard_errors) {{
                    model_binding::launch_{model}_{product}_cuda<
                        {template_arguments}
                    >(
                        device_models,
                        model_count,
                        device_prepared,
                        prepared_count,
                        device_products,
                        product_count,
                        context.construction,
                        context.result_count,
                        context.result_offset,
                        context.launch_result_count,
                        context.paths_per_price,
                        dt,
                        simulation_steps_per_day,
                        execution_profile.threads_per_block,
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
