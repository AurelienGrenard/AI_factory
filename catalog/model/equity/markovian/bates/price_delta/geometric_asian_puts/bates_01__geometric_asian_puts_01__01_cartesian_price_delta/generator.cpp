// Generated Cartesian-product bates geometric_asian_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/bates/product/geometric_asian_option_price_delta.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/geometric_asian_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/bates/parameters/bates_01.json", "datasets/product/geometric_asian_option/geometric_asian_options_01.json", "datasets/model/equity/markovian/bates/price_delta/geometric_asian_puts/bates_01__geometric_asian_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/bates/price_delta/geometric_asian_puts/bates_01__geometric_asian_puts_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/bates/price_delta/geometric_asian_puts/bates_01__geometric_asian_puts_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/bates/prices/geometric_asian_puts/bates_01__geometric_asian_puts_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "bates", "geometric_asian_option", ""},
        11668826724711268352ULL, model::equity::bates::load_models, product::load_geometric_asian_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::bates::launch_bates_geometric_asian_option_price_delta_cuda<OptionSide::put>(
                    host_models,
                    device_models,
                    model_count,
                    host_products,
                    device_products,
                    product_count,
                    PriceConstruction::CartesianProduct,
                    context.results,
                    context.offset,
                    context.count,
                    context.paths,
                    1.0f / 504.0f, 2U,
                    context.threads,
                    context.blocks,
                    context.seed,
                    context.bump,
                    prices,
                    price_errors,
                    deltas,
                    delta_errors);
        });
}
