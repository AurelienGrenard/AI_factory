// Generated Cartesian-product quadratic_rough_heston double_knock_out_option paired recipe using the existing host N-factor preparation.
#include "model/equity/rough/quadratic_rough_heston/product/double_knock_out_option_price_delta.cuh"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_preparation.hpp"
#include "product/double_knock_out_option/dataset.hpp"
#include "tools/pricing/prepared_price_delta_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pricing = offline::pricing;
    const datasets::PriceDeltaRecipe recipe{
        "datasets/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json", "datasets/product/double_knock_out_option/double_knock_out_options_01.json", "datasets/model/equity/rough/quadratic_rough_heston/price_delta/double_knock_out_calls/quadratic_rough_heston_01__double_knock_out_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/quadratic_rough_heston/price_delta/double_knock_out_calls/quadratic_rough_heston_01__double_knock_out_calls_01__01_cartesian_price_delta/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/quadratic_rough_heston/price_delta/double_knock_out_calls/quadratic_rough_heston_01__double_knock_out_calls_01__01_cartesian_price_delta.json", "catalog/model/equity/rough/quadratic_rough_heston/prices/double_knock_out_calls/quadratic_rough_heston_01__double_knock_out_calls_01__01_cartesian/generator.cpp", "centered_crn", .01, 2U, PriceConstruction::CartesianProduct};
    return pricing::generate_prepared_price_delta_dataset(
        recipe, {offline::cuda_tuning::PricingFamily::rough_n_factor, "quadratic_rough_heston", "double_knock_out_option", ""},
        11668828408338448384ULL, 7U, "7-factor Markovian lift", model::equity::quadratic_rough_heston::load_models, product::load_double_knock_out_options,
        [](const auto& models, float horizon) {
            return model::equity::quadratic_rough_heston::prepare_dynamics<7U>(models, horizon, 1.0f / 504.0f);
        },
        [](const auto* host_models, const auto* device_models, std::size_t model_count,
           const auto* device_prepared, std::size_t prepared_count,
           const auto* host_products, const auto* device_products, std::size_t product_count,
           const pricing::PriceDeltaLaunchContext& context,
           float* prices, float* price_errors, float* deltas, float* delta_errors) {
            model::equity::quadratic_rough_heston::launch_quadratic_rough_heston_double_knock_out_option_price_delta_cuda<OptionSide::call, 7U>(
                host_models, device_models, model_count, device_prepared, prepared_count,
                host_products, device_products, product_count, PriceConstruction::CartesianProduct,
                context.results, context.offset, context.count, context.paths,
                1.0f / 504.0f, 2U, context.threads, context.blocks, context.seed,
                context.bump, prices, price_errors, deltas, delta_errors);
        });
}
