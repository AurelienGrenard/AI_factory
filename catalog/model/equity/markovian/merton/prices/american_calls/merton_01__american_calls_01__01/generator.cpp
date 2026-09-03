// Generated Merton American-call price-dataset recipe.
#include "model/equity/markovian/merton/product/american_option.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "tools/pricing/american_option_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::merton;
    namespace pricing = offline::pricing;

    constexpr float day_fraction = 1.0f / 252.0f;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/markovian/merton/parameters/merton_01.json",
        "datasets/product/american_option/american_options_01.json",
        "datasets/model/equity/markovian/merton/prices/american_calls/"
        "merton_01__american_calls_01__01.json",
        "catalog/model/equity/markovian/merton/prices/american_calls/"
        "merton_01__american_calls_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/merton/prices/"
        "american_calls/merton_01__american_calls_01__01.json",
        "Exact Merton increments + Longstaff-Schwartz",
        PriceConstruction::Aligned,
    };
    const pricing::AmericanOptionProfile profile{
        1U << 20U,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseThreadsPerBlock,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseBlocksPerPrice,
        11668827407611068416ULL,
        "",
        "merton American call",
        "Spot and log-moneyness six-term basis",
        "FP64 normal equations and Cholesky on GPU",
        nlohmann::ordered_json::object(),
        nlohmann::ordered_json::array({"spot", "log_moneyness"}),
        nlohmann::ordered_json::array({"spot / strike", "log(spot / strike)"}),
        nlohmann::ordered_json::array({"1", "L1(spot / strike)", "L2(spot / strike)", "log(spot / strike)", "log(spot / strike)^2", "L1(spot / strike) * log(spot / strike)"}),
        true,
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
            return model_binding::launch_merton_american_option_cuda<
                OptionSide::call
            >(
                device_models,
                model_count,
                host_products,
                device_products,
                product_count,
                construction,
                result_count,
                paths_per_price,
                day_fraction,
                profile.threads_per_block,
                profile.blocks_per_price,
                profile.seed,
                device_prices,
                device_standard_errors
            );
        }
    );
}
