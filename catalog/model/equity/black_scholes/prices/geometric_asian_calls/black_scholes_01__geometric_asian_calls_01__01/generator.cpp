// Generated Black-Scholes geometric-asian-calls analytical price recipe.
#include "model/equity/markovian/black_scholes/geometric_asian_option.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/equity_price_generation.cuh"

#include <cstddef>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::black_scholes;
    namespace pricing = offline::pricing;

    const pricing::EquityPriceRecipe recipe{
        "datasets/model/equity/black_scholes/parameters/black_scholes_01.json",
        "datasets/product/equity/geometric_asian_options/geometric_asian_options_01.json",
        "datasets/model/equity/black_scholes/prices/geometric_asian_calls/black_scholes_01__geometric_asian_calls_01__01.json",
        "catalog/model/equity/black_scholes/prices/geometric_asian_calls/black_scholes_01__geometric_asian_calls_01__01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/black_scholes/prices/geometric_asian_calls/black_scholes_01__geometric_asian_calls_01__01.json",
        "Black-Scholes closed-form geometric-asian-calls",
        PriceConstruction::Aligned,
    };
    const pricing::AnalyticalProfile profile{1.0f / 504.0f, ::ai_factory::workbench::offline::cuda_tuning::kAnalyticalThreadsPerBlock, 2U};

    return pricing::generate_analytical_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        product::load_geometric_asian_options,
        [&](const auto& models, const auto& products,
            PriceConstruction construction,
            const pricing::AnalyticalProfile& execution_profile) {
            return pricing::execute_analytical(
                models,
                products,
                construction,
                execution_profile,
                [&](const auto* device_models, std::size_t model_count,
                    const auto* device_products, std::size_t product_count,
                    const pricing::AnalyticalLaunchContext& context,
                    float* device_prices) {
                    model_binding::launch_black_scholes_geometric_asian_option_cuda<OptionSide::call>(
                        device_models,
                        model_count,
                        device_products,
                        product_count,
                        context.construction,
                        context.result_count,
                        context.result_offset,
                        context.launch_result_count,
                        context.day_fraction,
                        context.simulation_steps_per_day,
                        context.threads_per_block,
                        context.block_count,
                        device_prices
                    );
                }
            );
        }
    );
}
