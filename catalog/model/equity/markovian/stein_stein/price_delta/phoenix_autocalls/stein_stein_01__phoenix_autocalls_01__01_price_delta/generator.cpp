// Generated aligned stein_stein phoenix_autocall price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/stein_stein/product/phoenix_autocall_price_delta.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "product/phoenix_autocall/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json", "datasets/product/phoenix_autocall/phoenix_autocalls_01.json", "datasets/model/equity/markovian/stein_stein/price_delta/phoenix_autocalls/stein_stein_01__phoenix_autocalls_01__01_price_delta.json", "catalog/model/equity/markovian/stein_stein/price_delta/phoenix_autocalls/stein_stein_01__phoenix_autocalls_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/stein_stein/price_delta/phoenix_autocalls/stein_stein_01__phoenix_autocalls_01__01_price_delta.json", "catalog/model/equity/markovian/stein_stein/prices/phoenix_autocalls/stein_stein_01__phoenix_autocalls_01__01/recipe.yaml", "centered_crn", .01, 2U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "stein_stein", "phoenix_autocall", ""},
        11668828056151130112ULL, model::equity::stein_stein::load_models, product::load_phoenix_autocalls,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::stein_stein::launch_stein_stein_phoenix_autocall_price_delta_cuda(
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
