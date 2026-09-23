// Generated aligned normal_inverse_gaussian range_accrual price-delta recipe; production paths/profile stay shared.
#include "model/equity/markovian/normal_inverse_gaussian/product/range_accrual_price_delta.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/pricing/equity_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json", "datasets/product/range_accrual/range_accruals_01.json", "datasets/model/equity/markovian/normal_inverse_gaussian/price_delta/range_accruals/normal_inverse_gaussian_01__range_accruals_01__01_price_delta.json", "catalog/model/equity/markovian/normal_inverse_gaussian/price_delta/range_accruals/normal_inverse_gaussian_01__range_accruals_01__01_price_delta/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/normal_inverse_gaussian/price_delta/range_accruals/normal_inverse_gaussian_01__range_accruals_01__01_price_delta.json", "catalog/model/equity/markovian/normal_inverse_gaussian/prices/range_accruals/normal_inverse_gaussian_01__range_accruals_01__01/recipe.yaml", "centered_crn", .01, 0U};
    return pricing::generate_equity_price_delta_dataset<true, false>(
        recipe, {offline::cuda_tuning::PricingFamily::equity_exact_mc, "normal_inverse_gaussian", "range_accrual", ""},
        11668827656719171584ULL, model::equity::normal_inverse_gaussian::load_models, product::load_range_accruals,
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            return model::equity::normal_inverse_gaussian::launch_normal_inverse_gaussian_range_accrual_price_delta_cuda(
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
