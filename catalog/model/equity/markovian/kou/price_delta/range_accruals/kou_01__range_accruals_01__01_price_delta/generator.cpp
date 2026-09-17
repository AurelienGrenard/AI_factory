// Generated aligned kou range_accrual price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/kou/product/range_accrual_price_delta.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/kou/parameters/kou_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/markovian/kou/price_delta/range_accruals/kou_01__range_accruals_01__01_price_delta.json", "catalog/model/equity/markovian/kou/price_delta/range_accruals/kou_01__range_accruals_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/kou/price_delta/range_accruals/kou_01__range_accruals_01__01_price_delta.json", "catalog/model/equity/markovian/kou/prices/range_accruals/kou_01__range_accruals_01__01/recipe.yaml", "centered_crn", .01, 0U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_exact_mc, "kou", "range_accrual", ""},
        11668827373251330048ULL, model::equity::kou::load_models, product::load_range_accruals,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::kou::launch_kou_range_accrual_price_delta_cuda(
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
