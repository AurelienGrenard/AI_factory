// Generated aligned kou athena_autocall price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/kou/product/athena_autocall_price_delta.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/kou/parameters/kou_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/markovian/kou/price_delta/athena_autocalls/kou_01__athena_autocalls_01__01_price_delta.json", "catalog/model/equity/markovian/kou/price_delta/athena_autocalls/kou_01__athena_autocalls_01__01_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/kou/price_delta/athena_autocalls/kou_01__athena_autocalls_01__01_price_delta.json", "catalog/model/equity/markovian/kou/prices/athena_autocalls/kou_01__athena_autocalls_01__01/generator.cpp", "centered_crn", .01, 0U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_exact_mc, "kou", "athena_autocall", ""},
        11668827291646951424ULL, model::equity::kou::load_models, product::load_athena_autocalls,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::kou::launch_kou_athena_autocall_price_delta_cuda(
                    host_models,
                    device_models,
                    model_count,
                    host_products,
                    device_products,
                    product_count,
                    PriceConstruction::Aligned,
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
