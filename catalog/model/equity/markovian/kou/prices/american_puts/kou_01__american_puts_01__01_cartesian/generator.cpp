// Generated Kou American-put price-dataset recipe.
#include "model/equity/markovian/kou/product/american_option.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/american_option_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::kou;
    namespace pricing = offline::pricing;

    constexpr float day_fraction = 1.0f / 252.0f;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/markovian/kou/parameters/kou_01.json",
        "datasets/product/american_option/american_options_01.json",
        "datasets/model/equity/markovian/kou/prices/american_puts/"
        "kou_01__american_puts_01__01_cartesian.json",
        "catalog/model/equity/markovian/kou/prices/american_puts/"
        "kou_01__american_puts_01__01_cartesian/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/kou/prices/"
        "american_puts/kou_01__american_puts_01__01_cartesian.json",
        "Exact Kou increments + Longstaff-Schwartz",
        PriceConstruction::CartesianProduct,
    };
    const pricing::AmericanOptionProfile profile{
        ::ai_factory::workbench::offline::cuda_tuning::kProductionPathsPerPrice,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseThreadsPerBlock,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseBlocksPerPrice,
        11668827270172114944ULL,
        "",
        "kou American put",
        "Spot and log-moneyness six-term basis",
        "FP64 normal equations and Cholesky on GPU, one normal-residual correction",
        nlohmann::ordered_json::object(),
        nlohmann::ordered_json::array({"spot", "log_moneyness"}),
        nlohmann::ordered_json::array({"spot / strike", "log(spot / strike)"}),
        nlohmann::ordered_json::array({"1", "L1(spot / strike)", "L2(spot / strike)", "log(spot / strike)", "log(spot / strike)^2", "L1(spot / strike) * log(spot / strike)"}),
        true,
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::equity_lsm, "kou", "american_option", ""},
    };

    return pricing::generate_american_option_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        [&](const offline::cuda_tuning::PricingLaunchPlan& plan,
            const auto* device_models, std::size_t model_count,
            const auto* host_products, const auto* device_products,
            std::size_t product_count, PriceConstruction construction,
            std::size_t result_count, std::size_t paths_per_price,
            float* device_prices, float* device_standard_errors) {
            return model_binding::launch_kou_american_option_cuda<
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
                day_fraction,
                plan.profile.threads_per_block,
                plan.profile.blocks_per_price,
                profile.seed,
                device_prices,
                device_standard_errors
            );
        }
    );
}
