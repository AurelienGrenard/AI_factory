// Generated Variance-Gamma european-calls price-dataset recipe.
#include "model/equity/markovian/variance_gamma/product/european_option.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/european_option/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::variance_gamma;
    namespace pricing = offline::pricing;

    constexpr float day_fraction = 1.0f / 252.0f;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json",
        "datasets/product/european_option/european_options_01.json",
        "datasets/model/equity/markovian/variance_gamma/prices/european_calls/variance_gamma_01__european_calls_01__01.json",
        "catalog/model/equity/markovian/variance_gamma/prices/european_calls/variance_gamma_01__european_calls_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/variance_gamma/prices/european_calls/variance_gamma_01__european_calls_01__01.json",
        "Exact Gamma subordination",
        PriceConstruction::Aligned,
    };
    const pricing::BatchedMonteCarloProfile profile{
        1'048'576U,
        4'096U,
        4'096U,
        ::ai_factory::workbench::offline::cuda_tuning::kMarkovianThreadsPerBlock,
        11668828159230345216ULL,
        "exact transition dates",
        nlohmann::ordered_json::object(),
    };

    return pricing::generate_monte_carlo_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        product::load_european_options,
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
                    model_binding::launch_variance_gamma_european_option_cuda<OptionSide::call>(
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
