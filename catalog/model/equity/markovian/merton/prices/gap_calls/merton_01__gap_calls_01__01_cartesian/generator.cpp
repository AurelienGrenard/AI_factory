// Generated Merton gap-calls price-dataset recipe.
#include "model/equity/markovian/merton/product/gap_option.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::merton;
    namespace pricing = offline::pricing;

    constexpr float day_fraction = 1.0f / 252.0f;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/markovian/merton/parameters/merton_01.json",
        "datasets/product/gap_option/gap_call_options_01.json",
        "datasets/model/equity/markovian/merton/prices/gap_calls/merton_01__gap_calls_01__01_cartesian.json",
        "catalog/model/equity/markovian/merton/prices/gap_calls/merton_01__gap_calls_01__01_cartesian/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/merton/prices/gap_calls/merton_01__gap_calls_01__01_cartesian.json",
        "Exact Merton increments",
        PriceConstruction::CartesianProduct,
    };
    const pricing::BatchedMonteCarloProfile profile{
        ::ai_factory::workbench::offline::cuda_tuning::kProductionPathsPerPrice,
        ::ai_factory::workbench::offline::cuda_tuning::kMonteCarloRowsPerLaunch,
        ::ai_factory::workbench::offline::cuda_tuning::kMonteCarloBlockCountLimit,
        ::ai_factory::workbench::offline::cuda_tuning::pricing_profile(::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::equity_exact_mc, "merton", "gap_option", ""}).threads_per_block,
        11668827484920479744ULL,
        "exact transition dates",
        nlohmann::ordered_json::object(),
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::equity_exact_mc, "merton", "gap_option", ""},
    };

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        [](const auto& path) { return product::load_gap_options(path, OptionSide::call); },
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
                    model_binding::launch_merton_gap_option_cuda<OptionSide::call>(
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
                        day_fraction,
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
