// Generated Bates phoenix-memory-autocalls price-dataset recipe.
#include "model/equity/markovian/bates/product/phoenix_memory_autocall.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/phoenix_memory_autocall/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::bates;
    namespace pricing = offline::pricing;

    constexpr float dt = 1.0f / 504.0f;
    constexpr std::uint32_t simulation_steps_per_day = 2U;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/markovian/bates/parameters/bates_01.json",
        "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json",
        "datasets/model/equity/markovian/bates/prices/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01_cartesian.json",
        "catalog/model/equity/markovian/bates/prices/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01_cartesian/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/bates/prices/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01_cartesian.json",
        "Andersen QE-M with compound-Poisson lognormal jumps",
        PriceConstruction::CartesianProduct,
    };
    const pricing::BatchedMonteCarloProfile profile{
        ::ai_factory::workbench::offline::cuda_tuning::kProductionPathsPerPrice,
        ::ai_factory::workbench::offline::cuda_tuning::kMonteCarloRowsPerLaunch,
        ::ai_factory::workbench::offline::cuda_tuning::kMonteCarloBlockCountLimit,
        ::ai_factory::workbench::offline::cuda_tuning::pricing_profile(::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "phoenix_memory_autocall", ""}).threads_per_block,
        11668826737596170240ULL,
        "1 / 504",
        nlohmann::ordered_json{{"simulation_steps_per_day", simulation_steps_per_day}},
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "phoenix_memory_autocall", ""},
    };

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        product::load_phoenix_memory_autocalls,
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
                    model_binding::launch_bates_phoenix_memory_autocall_cuda(
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
