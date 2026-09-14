// Generated Cartesian-product cev up_and_out_option price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/cev/product/up_and_out_option_price_delta.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/up_and_out_option/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/cev/parameters/cev_01.json", "datasets/product/up_and_out_option/up_and_out_options_01.json", "datasets/model/equity/markovian/cev/price_delta/up_and_out_calls/cev_01__up_and_out_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/cev/price_delta/up_and_out_calls/cev_01__up_and_out_calls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/cev/price_delta/up_and_out_calls/cev_01__up_and_out_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/cev/prices/up_and_out_calls/cev_01__up_and_out_calls_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "cev", "up_and_out_option", ""},
        11668826969524404224ULL, model::equity::cev::load_models, product::load_up_and_out_options,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::cev::launch_cev_up_and_out_option_price_delta_cuda<OptionSide::call>(
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
