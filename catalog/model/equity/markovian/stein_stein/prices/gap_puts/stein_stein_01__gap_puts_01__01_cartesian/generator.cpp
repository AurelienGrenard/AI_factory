// Generated Stein-Stein gap-puts price-dataset recipe.
#include "model/equity/markovian/stein_stein/product/gap_option.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/gap_option/dataset.hpp"
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
        "datasets/product/gap_option/gap_put_options_01.json",
        "datasets/model/equity/markovian/stein_stein/prices/gap_puts/stein_stein_01__gap_puts_01__01_cartesian.json",
        "catalog/model/equity/markovian/stein_stein/prices/gap_puts/stein_stein_01__gap_puts_01__01_cartesian/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/stein_stein/prices/gap_puts/stein_stein_01__gap_puts_01__01_cartesian.json",
        "exact OU volatility with log-spot Euler",
        PriceConstruction::CartesianProduct,
    };
    const pricing::BatchedMonteCarloProfile profile{
        ::ai_factory::workbench::offline::cuda_tuning::kProductionPathsPerPrice,
        ::ai_factory::workbench::offline::cuda_tuning::kMonteCarloRowsPerLaunch,
        ::ai_factory::workbench::offline::cuda_tuning::kMonteCarloBlockCountLimit,
        ::ai_factory::workbench::offline::cuda_tuning::pricing_profile(::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::equity_step_mc, "stein_stein", "gap_option", ""}).threads_per_block,
        11668828038971260928ULL,
        "1 / 504",
        nlohmann::ordered_json{{"simulation_steps_per_day", simulation_steps_per_day}},
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::equity_step_mc, "stein_stein", "gap_option", ""},
    };

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        [](const auto& path) { return product::load_gap_options(path, OptionSide::put); },
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
                    model_binding::launch_stein_stein_gap_option_cuda<OptionSide::put>(
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
