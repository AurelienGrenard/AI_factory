// Build one VarianceGamma Range Accrual price dataset from JSON inputs.
#include "model/equity/markovian/variance_gamma/range_accrual.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/range_accrual/dataset.hpp"
#include "tools/datasets/price_dataset.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Full input datasets and rule used to construct the price rows.
const std::filesystem::path model_dataset_path =
    "datasets/model/equity/variance_gamma/parameters/variance_gamma_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/equity/range_accruals/range_accruals_01.json";

constexpr ai_factory::workbench::PriceConstruction construction =
    ai_factory::workbench::PriceConstruction::Aligned;

// Numerical and CUDA configuration used by the pricing algorithm.
constexpr std::size_t monte_carlo_paths_per_price = 16'384U;
constexpr float day_fraction = 1.0f / 252.0f;
constexpr unsigned int threads_per_block = 512U;
constexpr std::size_t results_per_kernel_launch = 4'096U;
constexpr std::size_t block_count = 1'000U;
constexpr std::uint64_t seed = 900000001ULL;

// Artifact locations and descriptive metadata used after pricing.
const std::string random_generator = "Philox";
const std::filesystem::path dataset_path =
    "datasets/model/equity/variance_gamma/prices/range_accruals/"
    "variance_gamma_01__range_accruals_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/model/equity/variance_gamma/prices/range_accruals/"
    "variance_gamma_01__range_accruals_01__01/dataset.yaml";
const std::string url =
    "https://mlp.lpma.math.upmc.fr/DataCarlo/Assets/VarianceGamma/RangeAccrual/"
    "variance_gamma_01__range_accruals_01__01.json";
const std::string numerical_method = "Exact Gamma subordination";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;
    namespace variance_gamma = model::equity::variance_gamma;

    // 1. Load both datasets directly into contiguous FP32 vectors.
    const std::vector<variance_gamma::ModelParameters> models =
        variance_gamma::load_models(model_dataset_path);
    const std::vector<product::RangeAccrualParameters> products =
        product::load_range_accruals(product_dataset_path);

    // 2. Count the rows in the final price dataset.
    const std::size_t result_count = ai_factory::workbench::price_row_count(
        models.size(), products.size(), construction
    );

    // Bound each kernel duration while all result arrays remain on the GPU.
    const std::size_t kernel_launch_count =
        (result_count + results_per_kernel_launch - 1U)
        / results_per_kernel_launch;
    const std::size_t maximum_block_count = bounded_block_count(
        std::min(result_count, results_per_kernel_launch),
        block_count
    );

    // Allocate the two output arrays in host memory.

    // Declare the device pointers and CUDA events used below.
    const auto run = offline::cuda::run_monte_carlo(
        offline::cuda::inputs(models, products),
        result_count,
        [&](auto& execution) {
        const auto* device_models = execution.template input<0U>();
        const auto* device_products = execution.template input<1U>();
        auto* device_prices = execution.prices();
        auto* device_standard_errors = execution.standard_errors();
        const std::size_t warmup_row_count = std::min<std::size_t>(
            64U,
            std::min(models.size(), products.size())
        );
        const std::size_t warmup_block_count =
            ai_factory::workbench::bounded_block_count(
                warmup_row_count, block_count
            );
        variance_gamma::launch_variance_gamma_range_accrual_cuda(
            device_models,
            warmup_row_count,
            device_products,
            warmup_row_count,
            ai_factory::workbench::PriceConstruction::Aligned,
            warmup_row_count,
            0U,
            warmup_row_count,
            monte_carlo_paths_per_price,
            day_fraction,
            threads_per_block,
            warmup_block_count,
            seed,
            device_prices,
            device_standard_errors
        );

        },
        [&](auto& execution) {
        const auto* device_models = execution.template input<0U>();
        const auto* device_products = execution.template input<1U>();
        auto* device_prices = execution.prices();
        auto* device_standard_errors = execution.standard_errors();
        for (std::size_t result_offset = 0U;
             result_offset < result_count;
             result_offset += results_per_kernel_launch) {
            const std::size_t launch_result_count = std::min(
                results_per_kernel_launch,
                result_count - result_offset
            );
            const std::size_t launched_block_count =
                ai_factory::workbench::bounded_block_count(
                    launch_result_count, block_count
                );
            variance_gamma::launch_variance_gamma_range_accrual_cuda(
                device_models,
                models.size(),
                device_products,
                products.size(),
                construction,
                result_count,
                result_offset,
                launch_result_count,
                monte_carlo_paths_per_price,
                day_fraction,
                threads_per_block,
                launched_block_count,
                seed,
                device_prices,
                device_standard_errors
            );
        }


        }
    );

    // 5. Write the complete price dataset and catalog YAML.
    datasets::write_monte_carlo_price_dataset(
        model_dataset_path,
        product_dataset_path,
        construction,
        run.prices,
        run.standard_errors,
        random_generator,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        monte_carlo_paths_per_price,
        "",
        nlohmann::ordered_json{
            {"block_count", maximum_block_count},
            {"threads_per_block", threads_per_block},
            {"kernel_launch_count", kernel_launch_count},
            {"maximum_prices_per_block", 1U},
        },
        nlohmann::ordered_json{{
            "observation_schedule",
            {
                {"rule", "exact independent increments at product observation times"},
                {
                    "observation_count",
                    "round(maturity / observation_interval)"
                },
                {"increments_per_observation", 1U},
            }
        }},
        seed,
        run.wall_seconds,
        run.kernel_seconds
    );
    datasets::validate_price_dataset_file(dataset_path);
}
