// Build one Hull-White Nelson-Siegel caplet price dataset.
#include "common/check_cuda.cuh"
#include "model/hull_white/nelson_siegel/caplet.cuh"
#include "tools/datasets/dataset.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Full input datasets and rule used to construct the price rows.
const std::filesystem::path model_dataset_path =
    "datasets/model/hull_white/hull_white_01.json";
const std::filesystem::path curve_dataset_path =
    "datasets/curve/nelson_siegel/nelson_siegel_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/fixed_income/caplets/caplets_01.json";

constexpr ai_factory::workbench::datasets::PriceConstruction construction =
    ai_factory::workbench::datasets::PriceConstruction::Aligned;

// CUDA configuration for the one-thread-per-price analytical kernel.
constexpr unsigned int threads_per_block = 256U;
constexpr std::size_t results_per_kernel_launch = 1'000'000U;
constexpr std::size_t maximum_block_count = 4'096U;

// Artifact locations and descriptive metadata used after pricing.
const std::filesystem::path dataset_path =
    "datasets/price/hull_white/nelson_siegel/caplets/"
    "hull_white_01__nelson_siegel_01__caplets_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/price/hull_white/nelson_siegel/caplets/"
    "hull_white_01__nelson_siegel_01__caplets_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/price/hull_white/"
    "nelson_siegel/caplets/"
    "hull_white_01__nelson_siegel_01__caplets_01__01.json";
const std::string numerical_method =
    "Hull-White closed-form zero-coupon bond put";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;

    // 1. Load model, curve, and product rows into contiguous FP32 vectors.
    const std::vector<hull_white::HullWhiteModelParameters> models =
        hull_white::load_models(model_dataset_path);
    const std::vector<curve::nelson_siegel::NelsonSiegelParameters> curves =
        curve::nelson_siegel::load_curves(curve_dataset_path);
    const std::vector<product::CapletParameters> products =
        product::load_caplets(product_dataset_path);

    // 2. Count the rows in the final price dataset.
    const std::size_t result_count = datasets::price_row_count(
        models.size(), curves.size(), products.size(), construction
    );
    const std::size_t kernel_launch_count =
        (result_count - 1U) / results_per_kernel_launch + 1U;
    const auto block_count_for = [](std::size_t row_count) {
        const std::size_t required_blocks =
            (row_count - 1U) / threads_per_block + 1U;
        return std::min(required_blocks, maximum_block_count);
    };
    const std::size_t largest_launch_result_count = std::min(
        result_count, results_per_kernel_launch
    );
    const std::size_t launched_block_count =
        block_count_for(largest_launch_result_count);

    // Allocate the analytical price output in host memory.
    std::vector<float> prices(result_count);

    // Declare the four device arrays and CUDA timing events.
    hull_white::HullWhiteModelParameters* device_models = nullptr;
    curve::nelson_siegel::NelsonSiegelParameters* device_curves = nullptr;
    product::CapletParameters* device_products = nullptr;
    float* device_prices = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    // 3. Execute the complete GPU pipeline.
    const auto wall_start = std::chrono::system_clock::now();
    try {
        // Allocate model, curve, product, and price arrays on the GPU.
        check_cuda(
            cudaMalloc(
                &device_models,
                models.size() * sizeof(hull_white::HullWhiteModelParameters)
            ),
            "cudaMalloc Hull-White models"
        );
        check_cuda(
            cudaMalloc(
                &device_curves,
                curves.size()
                    * sizeof(curve::nelson_siegel::NelsonSiegelParameters)
            ),
            "cudaMalloc Nelson-Siegel curves"
        );
        check_cuda(
            cudaMalloc(
                &device_products,
                products.size() * sizeof(product::CapletParameters)
            ),
            "cudaMalloc caplets"
        );
        check_cuda(
            cudaMalloc(&device_prices, result_count * sizeof(float)),
            "cudaMalloc caplet prices"
        );

        // Copy all three input datasets from host memory to the GPU.
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(hull_white::HullWhiteModelParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Hull-White models"
        );
        check_cuda(
            cudaMemcpy(
                device_curves,
                curves.data(),
                curves.size()
                    * sizeof(curve::nelson_siegel::NelsonSiegelParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Nelson-Siegel curves"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(product::CapletParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy caplets"
        );

        // Warm up the specialized analytical kernel.
        const std::size_t warmup_row_count = std::min<std::size_t>(
            64U,
            std::min(models.size(), std::min(curves.size(), products.size()))
        );
        hull_white::nelson_siegel::
            launch_hull_white_nelson_siegel_caplet_cuda(
                device_models,
                warmup_row_count,
                device_curves,
                warmup_row_count,
                device_products,
                warmup_row_count,
                false,
                warmup_row_count,
                0U,
                warmup_row_count,
                threads_per_block,
                block_count_for(warmup_row_count),
                device_prices
            );
        check_cuda(cudaDeviceSynchronize(), "Hull-White caplet warmup");

        // Launch and time the complete production pricing kernel.
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");

        for (std::size_t result_offset = 0U;
             result_offset < result_count;
             result_offset += results_per_kernel_launch) {
            const std::size_t launch_result_count = std::min(
                results_per_kernel_launch, result_count - result_offset
            );
            hull_white::nelson_siegel::
                launch_hull_white_nelson_siegel_caplet_cuda(
                    device_models,
                    models.size(),
                    device_curves,
                    curves.size(),
                    device_products,
                    products.size(),
                    construction
                        == datasets::PriceConstruction::CartesianProduct,
                    result_count,
                    result_offset,
                    launch_result_count,
                    threads_per_block,
                    block_count_for(launch_result_count),
                    device_prices
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

        // Copy the analytical prices back to host memory.
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy caplet prices"
        );
    } catch (...) {
        // Release any CUDA resources acquired before the exception.
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        if (device_models != nullptr) cudaFree(device_models);
        if (device_curves != nullptr) cudaFree(device_curves);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    // 4. This generator prices once, so release every CUDA resource.
    check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
    check_cuda(cudaFree(device_models), "cudaFree Hull-White models");
    check_cuda(cudaFree(device_curves), "cudaFree Nelson-Siegel curves");
    check_cuda(cudaFree(device_products), "cudaFree caplets");
    check_cuda(cudaFree(device_prices), "cudaFree caplet prices");
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::system_clock::now() - wall_start
    ).count();

    // 5. Write the complete analytical price dataset and catalog YAML.
    datasets::write_analytical_price_dataset(
        model_dataset_path,
        curve_dataset_path,
        product_dataset_path,
        construction,
        prices,
        dataset_path,
        catalog_path,
        url,
        numerical_method,
        nlohmann::ordered_json{
            {"block_count", launched_block_count},
            {"threads_per_block", threads_per_block},
            {"kernel_launch_count", kernel_launch_count},
            {"work_distribution", "one price per thread, grid-stride loop"},
        },
        wall_seconds,
        kernel_seconds
    );
}
