// Generated SABR forward-start-puts price-dataset recipe.
#include "model/equity/markovian/sabr/forward_start_option.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::sabr;
    namespace pricing = offline::pricing;

    constexpr float dt = 1.0f / 504.0f;
    constexpr std::uint32_t simulation_steps_per_day = 2U;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/sabr/parameters/sabr_01.json",
        "datasets/product/equity/forward_start_options/forward_start_options_01.json",
        "datasets/model/equity/sabr/prices/forward_start_puts/sabr_01__forward_start_puts_01__01.json",
        "catalog/model/equity/sabr/prices/forward_start_puts/sabr_01__forward_start_puts_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/sabr/prices/forward_start_puts/sabr_01__forward_start_puts_01__01.json",
        "Lamperti SABR Euler",
        PriceConstruction::Aligned,
    };
    const pricing::BatchedMonteCarloProfile profile{
        1'048'576U,
        4'096U,
        4'096U,
        ::ai_factory::workbench::offline::cuda_tuning::kMarkovianThreadsPerBlock,
        900000001ULL,
        "1 / 504",
        nlohmann::ordered_json{{"simulation_steps_per_day", simulation_steps_per_day}},
    };

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        product::load_forward_start_options,
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::BatchedMonteCarloProfile& execution_profile) {
            return pricing::execute_batched_monte_carlo(
                models,
                products,
                construction,
                execution_profile,
                [&](const auto* device_models, std::size_t model_count,
                    const auto* device_products, std::size_t product_count,
                    const pricing::BatchedLaunchContext& context,
                    float* device_prices,
                    float* device_standard_errors) {
                    model_binding::launch_sabr_forward_start_option_cuda<OptionSide::put>(
                        device_models,
                        model_count,
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
