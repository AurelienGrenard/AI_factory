// Build one Black-Scholes straddles price dataset.
#include "common/check_cuda.cuh"
#include "model/equity/black_scholes/straddle.cuh"
#include "model/equity/black_scholes/dataset.hpp"
#include "product/straddle/dataset.hpp"
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Full input datasets and rule used to construct the price rows.
const std::filesystem::path model_dataset_path =
    "datasets/model/equity/black_scholes/parameters/black_scholes_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/equity/straddles/straddles_01.json";
constexpr ai_factory::workbench::datasets::PriceConstruction construction =
    ai_factory::workbench::datasets::PriceConstruction::Aligned;

// CUDA configuration for the one-thread-per-price analytical kernel.
constexpr float day_fraction = 1.0f / 252.0f;
constexpr unsigned int threads_per_block = 256U;

// Artifact locations and descriptive metadata used after pricing.
const std::filesystem::path dataset_path =
    "datasets/model/equity/black_scholes/prices/straddles/"
    "black_scholes_01__straddles_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/model/equity/black_scholes/prices/straddles/"
    "black_scholes_01__straddles_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/model/"
    "equity/black_scholes/prices/straddles/"
    "black_scholes_01__straddles_01__01.json";
const std::string numerical_method =
    "Black-Scholes closed-form straddle";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;
    namespace bs = black_scholes;

    // 1. Load model and product rows into contiguous FP32 vectors.
    const std::vector<bs::ModelParameters> models =
        bs::load_models(model_dataset_path);
    const std::vector<product::StraddleParameters> products =
        product::load_straddles(product_dataset_path);

    // 2. Count the rows in the final price dataset.
    const std::size_t result_count = datasets::price_row_count(
        models.size(), products.size(), construction
    );
    const auto block_count_for = [](std::size_t row_count) {
        return (row_count - 1U) / threads_per_block + 1U;
    };
    const std::size_t block_count = block_count_for(result_count);
    std::vector<float> prices(result_count);

    // Declare model, product, and output arrays with CUDA timing events.
    bs::ModelParameters* device_models = nullptr;
    product::StraddleParameters* device_products = nullptr;
    float* device_prices = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    // 3. Execute the complete GPU pipeline.
    const auto wall_start = std::chrono::steady_clock::now();
    try {
        check_cuda(
            cudaMalloc(&device_models, models.size() * sizeof(models.front())),
            "cudaMalloc Black-Scholes models"
        );
        check_cuda(
            cudaMalloc(
                &device_products, products.size() * sizeof(products.front())
            ),
            "cudaMalloc straddless"
        );
        check_cuda(
            cudaMalloc(&device_prices, result_count * sizeof(float)),
            "cudaMalloc Black-Scholes straddles prices"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Black-Scholes models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy straddless"
        );

        // Warm up the specialized analytical kernel.
        const std::size_t warmup_count = std::min<std::size_t>(
            64U, std::min(models.size(), products.size())
        );
        bs::launch_black_scholes_straddle_cuda(
            device_models,
            warmup_count,
            device_products,
            warmup_count,
            false,
            warmup_count,
            0U,
            warmup_count,
            day_fraction,
            threads_per_block,
            block_count_for(warmup_count),
            device_prices
        );
        check_cuda(cudaDeviceSynchronize(), "Black-Scholes straddles warmup");

        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");
        bs::launch_black_scholes_straddle_cuda(
            device_models,
            models.size(),
            device_products,
            products.size(),
            construction == datasets::PriceConstruction::CartesianProduct,
            result_count,
            0U,
            result_count,
            day_fraction,
            threads_per_block,
            block_count,
            device_prices
        );
        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");
        float kernel_milliseconds = 0.0f;
        check_cuda(
            cudaEventElapsedTime(
                &kernel_milliseconds, start_event, stop_event
            ),
            "cudaEventElapsedTime"
        );
        kernel_seconds = static_cast<double>(kernel_milliseconds) * 1.0e-3;
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy Black-Scholes straddles prices"
        );
    } catch (...) {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
    check_cuda(cudaFree(device_models), "cudaFree Black-Scholes models");
    check_cuda(cudaFree(device_products), "cudaFree straddless");
    check_cuda(cudaFree(device_prices), "cudaFree Black-Scholes straddles prices");
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();

    // 4. Write the complete analytical price dataset and catalog YAML.
    datasets::write_analytical_price_dataset(
        model_dataset_path,
        product_dataset_path,
        construction,
        prices,
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
        wall_seconds,
        kernel_seconds
    );
    datasets::validate_price_dataset_file(dataset_path);
}
