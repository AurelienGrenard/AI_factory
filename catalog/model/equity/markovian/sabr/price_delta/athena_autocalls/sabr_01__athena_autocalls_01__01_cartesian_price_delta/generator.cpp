// Generated Cartesian-product sabr athena_autocall price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/sabr/product/athena_autocall_price_delta.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/athena_autocall/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/athena_autocall/athena_autocalls_01.json", "datasets/model/equity/markovian/sabr/price_delta/athena_autocalls/sabr_01__athena_autocalls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/sabr/price_delta/athena_autocalls/sabr_01__athena_autocalls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/sabr/price_delta/athena_autocalls/sabr_01__athena_autocalls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/sabr/prices/athena_autocalls/sabr_01__athena_autocalls_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "athena_autocall", ""},
        11668827708258779136ULL, model::equity::sabr::load_models, product::load_athena_autocalls,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::sabr::launch_sabr_athena_autocall_price_delta_cuda(
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
