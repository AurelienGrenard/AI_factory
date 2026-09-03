// Generated {model_display} American-{side} price-dataset recipe.
#include "model/equity/markovian/{model}/product/american_option.cuh"
#include "model/equity/markovian/{model}/dataset.hpp"
#include "tools/pricing/american_option_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {{
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::{model};
    namespace pricing = offline::pricing;

{time_constants}
    const pricing::EquityPriceRecipe recipe{{
        "datasets/model/equity/markovian/{model}/parameters/{model}_01.json",
        "datasets/product/american_option/american_options_01.json",
        "datasets/model/equity/markovian/{model}/prices/american_{side}s/"
        "{database_id}.json",
        "catalog/model/equity/markovian/{model}/prices/american_{side}s/"
        "{database_id}/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/{model}/prices/"
        "american_{side}s/{database_id}.json",
        "{numerical_method} + Longstaff-Schwartz",
        PriceConstruction::Aligned,
    }};
    const pricing::AmericanOptionProfile profile{{
        1U << 20U,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseThreadsPerBlock,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseBlocksPerPrice,
        {seed}ULL,
        "{delta_t_description}",
        "{diagnostic_label}",
        "{regression_basis}",
        "{regression_precision}",
        {time_discretization},
        nlohmann::ordered_json::array({basis_state}),
        nlohmann::ordered_json::array({basis_normalization}),
        nlohmann::ordered_json::array({basis_functions}),
        {exact_exercise_dates},
    }};

    return pricing::generate_american_option_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        [&](const auto* device_models, std::size_t model_count,
            const auto* host_products, const auto* device_products,
            std::size_t product_count, PriceConstruction construction,
            std::size_t result_count, std::size_t paths_per_price,
            float* device_prices, float* device_standard_errors) {{
            return model_binding::launch_{model}_american_option_cuda<
                OptionSide::{side}
            >(
                device_models,
                model_count,
                host_products,
                device_products,
                product_count,
                construction,
                result_count,
                paths_per_price,
{time_arguments}                profile.threads_per_block,
                profile.blocks_per_price,
                profile.seed,
                device_prices,
                device_standard_errors
            );
        }}
    );
}}
