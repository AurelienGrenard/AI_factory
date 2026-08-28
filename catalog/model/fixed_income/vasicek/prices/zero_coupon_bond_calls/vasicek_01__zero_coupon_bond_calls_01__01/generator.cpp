// Build one Vasicek zero-coupon bond call price dataset.
#include "model/fixed_income/vasicek/dataset.hpp"
#include "model/fixed_income/vasicek/product/zero_coupon_bond_option.cuh"
#include "product/zero_coupon_bond_option/dataset.hpp"
#include "tools/datasets/price_dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Full input datasets and rule used to construct the price rows.
const std::filesystem::path model_dataset_path =
    "datasets/model/fixed_income/vasicek/parameters/vasicek_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/fixed_income/zero_coupon_bond_options/"
    "zero_coupon_bond_options_01.json";
constexpr ai_factory::workbench::PriceConstruction construction =
    ai_factory::workbench::PriceConstruction::Aligned;

// CUDA configuration for the one-thread-per-price analytical kernel.
constexpr unsigned int threads_per_block = 256U;
constexpr float day_fraction = 1.0f / 252.0f;

// Artifact locations and descriptive metadata used after pricing.
const std::filesystem::path dataset_path =
    "datasets/model/fixed_income/vasicek/prices/zero_coupon_bond_calls/"
    "vasicek_01__zero_coupon_bond_calls_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/model/fixed_income/vasicek/prices/zero_coupon_bond_calls/"
    "vasicek_01__zero_coupon_bond_calls_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/model/"
    "fixed_income/vasicek/prices/zero_coupon_bond_calls/"
    "vasicek_01__zero_coupon_bond_calls_01__01.json";
const std::string numerical_method =
    "Vasicek closed-form zero-coupon bond call";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;
    namespace vasicek = model::fixed_income::vasicek;

    // 1. Load model and product rows into contiguous FP32 vectors.
    const std::vector<vasicek::ModelParameters> models =
        vasicek::load_models(model_dataset_path);
    const std::vector<product::ZeroCouponBondOptionParameters> products =
        product::load_zero_coupon_bond_options(product_dataset_path);

    // 2. Count the rows in the final price dataset.
    const std::size_t result_count = ai_factory::workbench::price_row_count(
        models.size(), products.size(), construction
    );
    const auto block_count_for = [](std::size_t row_count) {
        return (row_count - 1U) / threads_per_block + 1U;
    };
    const std::size_t block_count = block_count_for(result_count);

    // Declare model, product, and output arrays with CUDA timing events.
    const auto run = offline::cuda::run_analytical(
        offline::cuda::inputs(models, products),
        result_count,
        [&](auto& execution) {
        const auto* device_models = execution.template input<0U>();
        const auto* device_products = execution.template input<1U>();
        auto* device_prices = execution.prices();
        const std::size_t warmup_count = std::min<std::size_t>(
            64U, std::min(models.size(), products.size())
        );
        vasicek::launch_vasicek_zero_coupon_bond_option_cuda<OptionSide::call>(
            device_models,
            warmup_count,
            device_products,
            warmup_count,
            ai_factory::workbench::PriceConstruction::Aligned,
            warmup_count,
            0U,
            warmup_count,
            day_fraction,
            threads_per_block,
            block_count_for(warmup_count),
            device_prices
        );

        },
        [&](auto& execution) {
        const auto* device_models = execution.template input<0U>();
        const auto* device_products = execution.template input<1U>();
        auto* device_prices = execution.prices();
        vasicek::launch_vasicek_zero_coupon_bond_option_cuda<OptionSide::call>(
            device_models,
            models.size(),
            device_products,
            products.size(),
            construction,
            result_count,
            0U,
            result_count,
            day_fraction,
            threads_per_block,
            block_count,
            device_prices
        );

        }
    );

    // 4. Write and structurally validate the analytical price dataset.
    datasets::write_analytical_price_dataset(
        model_dataset_path,
        product_dataset_path,
        construction,
        run.prices,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        nlohmann::ordered_json{
            {"block_count", block_count},
            {"threads_per_block", threads_per_block},
            {"kernel_launch_count", 1U},
            {"work_distribution", "one price per thread"},
        },
        run.wall_seconds,
        run.kernel_seconds
    );
    datasets::validate_price_dataset_file(dataset_path);
}
