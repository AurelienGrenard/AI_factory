// Generated Cartesian-product merton gap_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/merton/product/gap_option_price_delta.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "product/gap_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/merton/parameters/merton_01.json", "datasets/product/gap_option/gap_call_options_01.json", "datasets/model/equity/markovian/merton/price_delta/gap_calls/merton_01__gap_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/merton/price_delta/gap_calls/merton_01__gap_calls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/merton/price_delta/gap_calls/merton_01__gap_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/merton/prices/gap_calls/merton_01__gap_calls_01__01_cartesian/generator.cpp", "centered_crn", .01, 0U, PriceConstruction::CartesianProduct};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_exact_mc, "merton", "gap_option", ""},
        11668827484920479744ULL, model::equity::merton::load_models, [](const auto& path) { return product::load_gap_options(path, OptionSide::call); },
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::merton::launch_merton_gap_option_price_delta_cuda<OptionSide::call>(
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
                    1.0f / 252.0f,
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
