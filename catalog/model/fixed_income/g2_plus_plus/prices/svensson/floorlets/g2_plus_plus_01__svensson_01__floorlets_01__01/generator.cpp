// Generated Build one G2++ Svensson floorlet price dataset.
#include "model/fixed_income/g2_plus_plus/product/svensson/rate_option.cuh"
#include "curve/svensson/dataset.hpp"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "product/rate_option/dataset.hpp"
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
    "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json";
const std::filesystem::path curve_dataset_path =
    "datasets/curve/svensson/svensson_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/rate_option/rate_options_01.json";

constexpr ai_factory::workbench::PriceConstruction construction =
    ai_factory::workbench::PriceConstruction::Aligned;

// CUDA configuration for the one-thread-per-price analytical kernel.
constexpr unsigned int threads_per_block = 256U;
constexpr float day_fraction = 1.0f / 252.0f;

// Artifact locations and descriptive metadata used after pricing.
const std::filesystem::path dataset_path =
    "datasets/model/fixed_income/g2_plus_plus/prices/svensson/floorlets/"
    "g2_plus_plus_01__svensson_01__floorlets_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/model/fixed_income/g2_plus_plus/prices/svensson/floorlets/"
    "g2_plus_plus_01__svensson_01__floorlets_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/model/fixed_income/g2_plus_plus/prices/"
    "svensson/floorlets/"
    "g2_plus_plus_01__svensson_01__floorlets_01__01.json";
const std::string numerical_method =
    "G2++ closed-form zero-coupon bond call";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;
    namespace g2pp = model::fixed_income::g2_plus_plus;
    namespace fitted = g2pp::svensson;

    // 1. Load model, curve, and product rows into contiguous FP32 vectors.
    const std::vector<g2pp::ModelParameters> models =
        g2pp::load_models(model_dataset_path);
    const std::vector<curve::svensson::SvenssonParameters> curves =
        curve::svensson::load_curves(curve_dataset_path);
    const std::vector<product::RateOptionParameters> products =
        product::load_rate_options(product_dataset_path);

    // 2. Count the rows in the final price dataset.
    const std::size_t result_count = ai_factory::workbench::price_row_count(
        models.size(), curves.size(), products.size(), construction
    );
    const auto block_count_for = [](std::size_t row_count) {
        return (row_count - 1U) / threads_per_block + 1U;
    };
    const std::size_t block_count = block_count_for(result_count);

    // Allocate the analytical price output in host memory.

    // Declare the four device arrays and CUDA timing events.
    const auto run = offline::cuda::run_analytical(
        offline::cuda::inputs(models, curves, products),
        result_count,
        [&](auto& execution) {
        const auto* device_models = execution.template input<0U>();
        const auto* device_curves = execution.template input<1U>();
        const auto* device_products = execution.template input<2U>();
        auto* device_prices = execution.prices();
        const std::size_t warmup_count = std::min<std::size_t>(
            64U,
            std::min(models.size(), std::min(curves.size(), products.size()))
        );
        fitted::launch_g2_plus_plus_svensson_rate_option_cuda<OptionSide::put>(
            device_models,
            warmup_count,
            device_curves,
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
        const auto* device_curves = execution.template input<1U>();
        const auto* device_products = execution.template input<2U>();
        auto* device_prices = execution.prices();
        fitted::launch_g2_plus_plus_svensson_rate_option_cuda<OptionSide::put>(
            device_models,
            models.size(),
            device_curves,
            curves.size(),
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

    // 5. Write the complete analytical price dataset and catalog YAML.
    datasets::write_analytical_price_dataset(
        model_dataset_path,
        curve_dataset_path,
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
