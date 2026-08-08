// Build one CEV Down-and-out-put price dataset from JSON inputs.
#include "common/check_cuda.cuh"
#include "model/equity/cev/down_and_out_option.cuh"
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
    "datasets/model/equity/cev/cev_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/equity/down_and_out_options/down_and_out_options_01.json";

constexpr ai_factory::workbench::datasets::PriceConstruction construction =
    ai_factory::workbench::datasets::PriceConstruction::Aligned;

// Numerical and CUDA configuration used by the pricing algorithm.
constexpr std::size_t monte_carlo_paths_per_price = 16'384U;
constexpr float target_dt = 1.0f / 252.0f;
constexpr unsigned int threads_per_block = 512U;
constexpr std::size_t results_per_kernel_launch = 4'096U;
constexpr std::size_t block_count = 4'096U;
constexpr std::uint64_t seed = 900000001ULL;

// Artifact locations and descriptive metadata used after pricing.
const std::string target_dt_description = "1 / 252";
const std::string random_generator = "Philox";
const std::filesystem::path dataset_path =
    "datasets/price/equity/cev/down_and_out_puts/"
    "cev_01__down_and_out_puts_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/price/equity/cev/down_and_out_puts/"
    "cev_01__down_and_out_puts_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/price/equity/cev/down_and_out_puts/"
    "cev_01__down_and_out_puts_01__01.json";
const std::string numerical_method = "absorbed Milstein";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;

    // 1. Load both datasets directly into contiguous FP32 vectors.
    const std::vector<cev::CevModelParameters> models =
        cev::load_models(model_dataset_path);
    const std::vector<product::DownAndOutOptionParameters> products =
        product::load_down_and_out_options(product_dataset_path);

    // 2. Count the rows in the final price dataset.
    const std::size_t result_count = datasets::price_row_count(
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
    std::vector<float> prices(result_count);
    std::vector<float> standard_errors(result_count);

    // Declare the device pointers and CUDA events used below.
    cev::CevModelParameters* device_models = nullptr;
    product::DownAndOutOptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    float* device_standard_errors = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    // 3. Execute the complete GPU pipeline.
    const auto wall_start = std::chrono::steady_clock::now();
    try {
        // Allocate the model, product, price, and error arrays on the GPU.
        check_cuda(
            cudaMalloc(
                &device_models,
                models.size() * sizeof(cev::CevModelParameters)
            ),
            "cudaMalloc CEV models"
        );
        check_cuda(
            cudaMalloc(
                &device_products,
                products.size() * sizeof(product::DownAndOutOptionParameters)
            ),
            "cudaMalloc Down-and-out puts"
        );
        check_cuda(
            cudaMalloc(
                &device_prices,
                result_count * sizeof(float)
            ),
            "cudaMalloc CEV put prices"
        );
        check_cuda(
            cudaMalloc(
                &device_standard_errors,
                result_count * sizeof(float)
            ),
            "cudaMalloc CEV put standard errors"
        );

        // Copy the model and product parameters from host memory to the GPU.
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(cev::CevModelParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy CEV models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(product::DownAndOutOptionParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Down-and-out puts"
        );

        // Warm up the specialized kernel and establish normal GPU clocks.
        const std::size_t warmup_row_count = std::min<std::size_t>(
            64U,
            std::min(models.size(), products.size())
        );
        const std::size_t warmup_block_count =
            ai_factory::workbench::bounded_block_count(
                warmup_row_count, block_count
            );
        cev::launch_cev_down_and_out_option_cuda<OptionSide::put>(
            device_models,
            warmup_row_count,
            device_products,
            warmup_row_count,
            false,
            warmup_row_count,
            0U,
            warmup_row_count,
            monte_carlo_paths_per_price,
            target_dt,
            threads_per_block,
            warmup_block_count,
            seed,
            device_prices,
            device_standard_errors
        );
        check_cuda(cudaDeviceSynchronize(), "CEV put warmup");

        // Launch and time the complete production pricing kernel.
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");

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
            cev::launch_cev_down_and_out_option_cuda<OptionSide::put>(
                device_models,
                models.size(),
                device_products,
                products.size(),
                construction == datasets::PriceConstruction::CartesianProduct,
                result_count,
                result_offset,
                launch_result_count,
                monte_carlo_paths_per_price,
                target_dt,
                threads_per_block,
                launched_block_count,
                seed,
                device_prices,
                device_standard_errors
            );
        }

        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
        float kernel_milliseconds = 0.0f;
        check_cuda(
            cudaEventElapsedTime(
                &kernel_milliseconds, start_event, stop_event
            ),
            "cudaEventElapsedTime"
        );
        kernel_seconds =
            static_cast<double>(kernel_milliseconds) * 1.0e-3;

        // Copy the final prices and standard errors back to host memory.
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy CEV put prices"
        );
        check_cuda(
            cudaMemcpy(
                standard_errors.data(),
                device_standard_errors,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy CEV put standard errors"
        );
    } catch (...) {
        // Release any CUDA resources acquired before the exception.
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        if (device_standard_errors != nullptr) cudaFree(device_standard_errors);
        throw;
    }

    // 4. This generator prices once, so every CUDA resource is released now.
    check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
    check_cuda(cudaFree(device_models), "cudaFree CEV models");
    check_cuda(cudaFree(device_products), "cudaFree Down-and-out puts");
    check_cuda(cudaFree(device_prices), "cudaFree CEV put prices");
    check_cuda(
        cudaFree(device_standard_errors),
        "cudaFree CEV put standard errors"
    );
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();

    // 5. Write the complete price dataset and catalog YAML.
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
        nlohmann::ordered_json{
            {"block_count", maximum_block_count},
            {"threads_per_block", threads_per_block},
            {"kernel_launch_count", kernel_launch_count},
            {"maximum_prices_per_block", 1U},
        },
        nlohmann::ordered_json::object(),
        seed,
        wall_seconds,
        kernel_seconds
    );
    datasets::validate_price_dataset_file(dataset_path);
}
