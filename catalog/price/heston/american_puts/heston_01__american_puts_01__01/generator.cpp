// Build one Heston American-put price dataset from JSON inputs.
#include "common/check_cuda.cuh"
#include "model/heston/american_option.cuh"
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Full input datasets and rule used to construct the price rows.
const std::filesystem::path model_dataset_path =
    "datasets/model/heston/heston_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/equity/american_options/american_options_01.json";

constexpr ai_factory::workbench::datasets::PriceConstruction construction =
    ai_factory::workbench::datasets::PriceConstruction::Aligned;

// Numerical and CUDA configuration used by the pricing algorithm.
constexpr std::size_t monte_carlo_paths_per_price = 1U << 20U;
constexpr float target_dt = 1.0f / 252.0f;
constexpr unsigned int threads_per_block = 128U;
constexpr std::size_t blocks_per_price = 128U;
constexpr std::uint64_t seed = 900000001ULL;

// Artifact locations and descriptive metadata used after pricing.
const std::string target_dt_description = "1 / 252";
const std::string random_generator = "Philox";
const std::filesystem::path dataset_path =
    "datasets/price/heston/american_puts/"
    "heston_01__american_puts_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/price/heston/american_puts/"
    "heston_01__american_puts_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/price/heston/american_puts/"
    "heston_01__american_puts_01__01.json";
const std::string numerical_method = "Andersen QE-M + Longstaff-Schwartz";
const std::string regression_basis = "Two-factor Laguerre degree 2";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;

    // 1. Load both datasets directly into contiguous FP32 vectors.
    const std::vector<heston::HestonModelParameters> models =
        heston::load_models(model_dataset_path);
    const std::vector<product::AmericanOptionParameters> products =
        product::load_american_options(product_dataset_path);

    // 2. Count the rows in the final price dataset.
    const std::size_t result_count = datasets::price_row_count(
        models.size(), products.size(), construction
    );
    // Allocate the two output arrays in host memory.
    std::vector<float> prices(result_count);
    std::vector<float> standard_errors(result_count);

    // Declare the persistent device arrays used by every memory-aware batch.
    heston::HestonModelParameters* device_models = nullptr;
    product::AmericanOptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    float* device_standard_errors = nullptr;
    longstaff_schwartz::LaunchResult execution{};
    double wall_seconds = 0.0;

    // 3. Allocate and load the persistent input and output arrays.
    std::chrono::steady_clock::time_point wall_start;
    try {
        // Allocate the model, product, price, and error arrays on the GPU.
        check_cuda(
            cudaMalloc(
                &device_models,
                models.size() * sizeof(heston::HestonModelParameters)
            ),
            "cudaMalloc Heston models"
        );
        check_cuda(
            cudaMalloc(
                &device_products,
                products.size() * sizeof(product::AmericanOptionParameters)
            ),
            "cudaMalloc American puts"
        );
        check_cuda(
            cudaMalloc(
                &device_prices,
                result_count * sizeof(float)
            ),
            "cudaMalloc Heston American put prices"
        );
        check_cuda(
            cudaMalloc(
                &device_standard_errors,
                result_count * sizeof(float)
            ),
            "cudaMalloc Heston American put standard errors"
        );

        // Copy the model and product parameters from host memory to the GPU.
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(heston::HestonModelParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Heston models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(product::AmericanOptionParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy American puts"
        );

        // Warm one small row so JIT loading and GPU clocks stay out of timing.
        constexpr std::size_t warmup_paths = 4'096U;
        heston::launch_heston_american_option_cuda<OptionSide::put>(
            device_models,
            1U,
            products.data(),
            device_products,
            1U,
            false,
            1U,
            std::min(monte_carlo_paths_per_price, warmup_paths),
            target_dt,
            threads_per_block,
            blocks_per_price,
            seed,
            device_prices,
            device_standard_errors
        );
        check_cuda(cudaDeviceSynchronize(), "Heston American-put warmup");

        // Plan the reusable workspace and time every production batch.
        wall_start = std::chrono::steady_clock::now();
        execution = heston::launch_heston_american_option_cuda<OptionSide::put>(
            device_models,
            models.size(),
            products.data(),
            device_products,
            products.size(),
            construction == datasets::PriceConstruction::CartesianProduct,
            result_count,
            monte_carlo_paths_per_price,
            target_dt,
            threads_per_block,
            blocks_per_price,
            seed,
            device_prices,
            device_standard_errors
        );

        // Copy the final prices and standard errors back to host memory.
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy Heston American put prices"
        );
        check_cuda(
            cudaMemcpy(
                standard_errors.data(),
                device_standard_errors,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy Heston American put standard errors"
        );
        wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start
        ).count();
    } catch (...) {
        // Release any CUDA resources acquired before the exception.
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        if (device_standard_errors != nullptr) cudaFree(device_standard_errors);
        throw;
    }

    // 4. This generator prices once, so every CUDA resource is released now.
    check_cuda(cudaFree(device_models), "cudaFree Heston models");
    check_cuda(cudaFree(device_products), "cudaFree American puts");
    check_cuda(cudaFree(device_prices), "cudaFree Heston American put prices");
    check_cuda(
        cudaFree(device_standard_errors),
        "cudaFree Heston American put standard errors"
    );
    // 5. Write the complete price dataset and catalog YAML.
    const nlohmann::ordered_json cuda_execution = {
        {"threads_per_block", threads_per_block},
        {"blocks_per_price", execution.blocks_per_price},
    };
    const nlohmann::ordered_json catalog_sections = {
        {"time_grid", {
            {"target_dt", target_dt_description},
            {
                "initial_stub",
                {
                    {"step_count", "ceil(first_exercise_time / target_dt)"},
                    {"effective_dt", "first_exercise_time / step_count"},
                }
            },
            {
                "regular_exercise_interval",
                {
                    {"step_count", "ceil(exercise_interval / target_dt)"},
                    {"effective_dt", "exercise_interval / step_count"},
                }
            },
        }},
        {"outputs", {
            {
                "price",
                {{"estimator", "Longstaff-Schwartz discounted cashflow mean"}}
            },
            {
                "standard_error",
                {
                    {
                        "estimator",
                        "Conditional standard error of discounted policy cashflows"
                    }
                }
            },
        }},
        {"exercise_policy", {
            {"method", "Longstaff-Schwartz"},
            {"regression_basis", regression_basis},
            {"exercise_dates", "time zero and maturity-anchored product dates"},
            {"regression_target", "realized future cashflow discounted one exercise interval"},
            {"in_the_money_paths_only", true},
            {"solver", "Cholesky on GPU"},
            {"basis", {
                {"state", {"spot", "instantaneous_variance"}},
                {"normalization", {"spot / strike", "variance / theta"}},
                {
                    "functions",
                    {
                        "1",
                        "L1(spot / strike)",
                        "L2(spot / strike)",
                        "variance / theta",
                        "(variance / theta)^2",
                        "L1(spot / strike) * variance / theta",
                    }
                },
                {
                    "regularization",
                    {{"ridge", "1e-10 * trace(G) / basis_size"}}
                },
            }},
        }},
    };
    datasets::write_monte_carlo_price_dataset(
        model_dataset_path,
        product_dataset_path,
        construction,
        prices,
        standard_errors,
        random_generator,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        monte_carlo_paths_per_price,
        target_dt_description,
        cuda_execution,
        catalog_sections,
        seed,
        wall_seconds,
        execution.kernel_seconds
    );
    datasets::validate_price_dataset_file(dataset_path);
}
