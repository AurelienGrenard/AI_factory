// Generated Stein-Stein digital-puts price-dataset recipe.
#include "model/equity/markovian/stein_stein/product/digital_option.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/digital_option/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::stein_stein;
    namespace pricing = offline::pricing;

    constexpr float dt = 1.0f / 504.0f;
    constexpr std::uint32_t simulation_steps_per_day = 2U;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json",
        "datasets/product/digital_option/digital_options_01.json",
        "datasets/model/equity/markovian/stein_stein/prices/digital_puts/stein_stein_01__digital_puts_01__01.json",
        "catalog/model/equity/markovian/stein_stein/prices/digital_puts/stein_stein_01__digital_puts_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/stein_stein/prices/digital_puts/stein_stein_01__digital_puts_01__01.json",
        "exact OU volatility with log-spot Euler",
        PriceConstruction::Aligned,
    };
    const pricing::BatchedMonteCarloProfile profile{
        1'048'576U,
        4'096U,
        4'096U,
        ::ai_factory::workbench::offline::cuda_tuning::kMarkovianThreadsPerBlock,
        11668827996021587968ULL,
        "1 / 504",
        nlohmann::ordered_json{{"simulation_steps_per_day", simulation_steps_per_day}},
    };

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        product::load_digital_options,
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::BatchedMonteCarloProfile& execution_profile) {
            return pricing::execute_batched_monte_carlo(
                models,
                products,
                construction,
                execution_profile,
                [&](const auto* device_models, std::size_t model_count,
                    const auto* host_products,
                    const auto* device_products, std::size_t product_count,
                    const pricing::BatchedLaunchContext& context,
                    float* device_prices,
                    float* device_standard_errors) {
                    model_binding::launch_stein_stein_digital_option_cuda<OptionSide::put>(
                        device_models,
                        model_count,
                        host_products,
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
                }
            );
        }
    );
}
