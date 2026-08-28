// Generated Normal-Inverse-Gaussian cliquets price-dataset recipe.
#include "model/equity/markovian/normal_inverse_gaussian/product/cliquet.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/cliquet/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::normal_inverse_gaussian;
    namespace pricing = offline::pricing;

    constexpr float day_fraction = 1.0f / 252.0f;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json",
        "datasets/product/equity/cliquets/cliquets_01.json",
        "datasets/model/equity/normal_inverse_gaussian/prices/cliquets/normal_inverse_gaussian_01__cliquets_01__01.json",
        "catalog/model/equity/normal_inverse_gaussian/prices/cliquets/normal_inverse_gaussian_01__cliquets_01__01/dataset.yaml",
        "https://mlp.lpma.math.upmc.fr/DataCarlo/Assets/NormalInverseGaussian/Cliquet/normal_inverse_gaussian_01__cliquets_01__01.json",
        "Exact inverse-Gaussian subordination",
        PriceConstruction::Aligned,
    };
    const pricing::BatchedMonteCarloProfile profile{
        1'048'576U,
        4'096U,
        4'096U,
        ::ai_factory::workbench::offline::cuda_tuning::kMarkovianCompactThreadsPerBlock,
        900000001ULL,
        "exact transition dates",
        nlohmann::ordered_json::object(),
    };

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        product::load_cliquets,
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
                    model_binding::launch_normal_inverse_gaussian_cliquet_cuda(
                        device_models,
                        model_count,
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
