// Generated Rough-Heston asian-puts N-factor price recipe.
#include "model/equity/rough/rough_heston/product/asian_option.cuh"
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"
#include "product/asian_option/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::rough_heston;
    namespace pricing = offline::pricing;

    constexpr std::size_t factor_count = 7U;
    constexpr float day_fraction = 1.0f / 252.0f;
    constexpr float dt = 1.0f / 504.0f;
    constexpr std::uint32_t simulation_steps_per_day = 2U;
    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json",
        "datasets/product/asian_option/asian_options_01.json",
        "datasets/model/equity/rough/rough_heston/prices/asian_puts/rough_heston_01__asian_puts_01__01.json",
        "catalog/model/equity/rough/rough_heston/prices/asian_puts/rough_heston_01__asian_puts_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_heston/prices/asian_puts/rough_heston_01__asian_puts_01__01.json",
        "7-factor Markovian lift",
        PriceConstruction::Aligned,
    };
    const pricing::BatchedMonteCarloProfile profile{
        1'048'576U,
        4'096U,
        4'096U,
        ::ai_factory::workbench::offline::cuda_tuning::kNFactorThreadsPerBlock,
        11668828644561649664ULL,
        "1 / 504",
        nlohmann::ordered_json{
            {"simulation_steps_per_day", simulation_steps_per_day},
            {"factor_count", factor_count},
        },
    };

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        product::load_asian_options,
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::BatchedMonteCarloProfile& execution_profile) {
            float approximation_horizon = day_fraction;
            for (const auto& product : products) {
                approximation_horizon = std::max(
                    approximation_horizon,
                    static_cast<float>(product.maturity_days) * day_fraction
                );
            }
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
                    const auto* host_products,
                    const auto* device_products, std::size_t product_count,
                    const pricing::BatchedLaunchContext& context,
                    float* device_prices,
                    float* device_standard_errors) {
                    model_binding::launch_rough_heston_asian_option_cuda<
                        OptionSide::put, factor_count
                    >(
                        device_models,
                        model_count,
                        device_prepared,
                        prepared_count,
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
