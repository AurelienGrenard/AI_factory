// Generated Cartesian-product sabr up_one_touch price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/sabr/product/up_one_touch_price_delta.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/sabr/parameters/sabr_01.json", "datasets/product/up_one_touch/up_one_touches_01.json", "datasets/model/equity/markovian/sabr/price_delta/up_one_touches/sabr_01__up_one_touches_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/sabr/price_delta/up_one_touches/sabr_01__up_one_touches_01__01_cartesian_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/sabr/price_delta/up_one_touches/sabr_01__up_one_touches_01__01_cartesian_price_delta.json", "catalog/model/equity/markovian/sabr/prices/up_one_touches/sabr_01__up_one_touches_01__01_cartesian/recipe.yaml", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_step_mc, "sabr", "up_one_touch", ""},
        11668827811337994240ULL, model::equity::sabr::load_models, product::load_up_one_touches,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::sabr::launch_sabr_up_one_touch_price_delta_cuda(
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
