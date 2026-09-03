// Generated Schobel-Zhu American-put price-dataset recipe.
#include "model/equity/markovian/schobel_zhu/product/american_option.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "tools/pricing/american_option_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::schobel_zhu;
    namespace pricing = offline::pricing;

    constexpr float dt = 1.0f / 504.0f;
    constexpr std::uint32_t simulation_steps_per_day = 2U;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json",
        "datasets/product/american_option/american_options_01.json",
        "datasets/model/equity/markovian/schobel_zhu/prices/american_puts/"
        "schobel_zhu_01__american_puts_01__01.json",
        "catalog/model/equity/markovian/schobel_zhu/prices/american_puts/"
        "schobel_zhu_01__american_puts_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/schobel_zhu/prices/"
        "american_puts/schobel_zhu_01__american_puts_01__01.json",
        "exact OU factor with log-spot Euler + Longstaff-Schwartz",
        PriceConstruction::Aligned,
    };
    const pricing::AmericanOptionProfile profile{
        1U << 20U,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseThreadsPerBlock,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseBlocksPerPrice,
        11668827828517863424ULL,
        "1 / 504",
        "schobel-zhu American put",
        "Two-factor Laguerre degree 2",
        "FP64 normal equations and Cholesky on GPU",
        nlohmann::ordered_json{{"simulation_steps_per_day", simulation_steps_per_day}},
        nlohmann::ordered_json::array({"spot", "volatility"}),
        nlohmann::ordered_json::array({"spot / strike", "volatility / long_run_volatility"}),
        nlohmann::ordered_json::array({"1", "L1(spot / strike)", "L2(spot / strike)", "volatility / long_run_volatility", "(volatility / long_run_volatility)^2", "L1(spot / strike) * volatility / long_run_volatility"}),
        false,
    };

    return pricing::generate_american_option_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        [&](const auto* device_models, std::size_t model_count,
            const auto* host_products, const auto* device_products,
            std::size_t product_count, PriceConstruction construction,
            std::size_t result_count, std::size_t paths_per_price,
            float* device_prices, float* device_standard_errors) {
            return model_binding::launch_schobel_zhu_american_option_cuda<
                OptionSide::put
            >(
                device_models,
                model_count,
                host_products,
                device_products,
                product_count,
                construction,
                result_count,
                paths_per_price,
                dt,
                simulation_steps_per_day,
                profile.threads_per_block,
                profile.blocks_per_price,
                profile.seed,
                device_prices,
                device_standard_errors
            );
        }
    );
}
