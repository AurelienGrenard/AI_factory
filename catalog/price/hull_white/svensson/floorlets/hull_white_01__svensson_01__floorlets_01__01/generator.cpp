// Build one Hull-White Svensson floorlet price dataset.
#include "common/check_cuda.cuh"
#include "model/hull_white/svensson/floorlet.cuh"
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
    "datasets/model/hull_white/hull_white_01.json";
const std::filesystem::path curve_dataset_path =
    "datasets/curve/svensson/svensson_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/fixed_income/floorlets/floorlets_01.json";

constexpr ai_factory::workbench::datasets::PriceConstruction construction =
    ai_factory::workbench::datasets::PriceConstruction::Aligned;

// CUDA configuration for the one-thread-per-price analytical kernel.
constexpr unsigned int threads_per_block = 256U;

// Artifact locations and descriptive metadata used after pricing.
const std::filesystem::path dataset_path =
    "datasets/price/hull_white/svensson/floorlets/"
    "hull_white_01__svensson_01__floorlets_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/price/hull_white/svensson/floorlets/"
    "hull_white_01__svensson_01__floorlets_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/price/hull_white/"
    "svensson/floorlets/"
    "hull_white_01__svensson_01__floorlets_01__01.json";
const std::string numerical_method =
    "Hull-White closed-form zero-coupon bond call";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;
    namespace hw = model::hull_white;
    namespace fitted = hw::svensson;

    // 1. Load model, curve, and product rows into contiguous FP32 vectors.
    const std::vector<hw::HullWhiteModelParameters> models =
        hw::load_models(model_dataset_path);
    const std::vector<curve::svensson::SvenssonParameters> curves =
        curve::svensson::load_curves(curve_dataset_path);
    const std::vector<product::FloorletParameters> products =
        product::load_floorlets(product_dataset_path);

    // 2. Count the rows in the final price dataset.
    const std::size_t result_count = datasets::price_row_count(
        models.size(), curves.size(), products.size(), construction
    );
    const auto block_count_for = [](std::size_t row_count) {
        return (row_count - 1U) / threads_per_block + 1U;
    };
    const std::size_t block_count = block_count_for(result_count);

    // Allocate the analytical price output in host memory.
    std::vector<float> prices(result_count);

    // Declare the four device arrays and CUDA timing events.
    hw::HullWhiteModelParameters* device_models = nullptr;
    curve::svensson::SvenssonParameters* device_curves = nullptr;
    product::FloorletParameters* device_products = nullptr;
    float* device_prices = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    // 3. Execute the complete GPU pipeline.
    const auto wall_start = std::chrono::system_clock::now();
    try {
        check_cuda(
            cudaMalloc(&device_models, models.size() * sizeof(models.front())),
            "cudaMalloc Hull-White models"
        );
        check_cuda(
            cudaMalloc(
                &device_curves,
                curves.size() * sizeof(curves.front())
            ),
            "cudaMalloc Svensson curves"
        );
        check_cuda(
            cudaMalloc(
                &device_products,
                products.size() * sizeof(products.front())
            ),
            "cudaMalloc floorlets"
        );
        check_cuda(
            cudaMalloc(&device_prices, result_count * sizeof(float)),
            "cudaMalloc floorlet prices"
        );

        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Hull-White models"
        );
        check_cuda(
            cudaMemcpy(
                device_curves,
                curves.data(),
                curves.size() * sizeof(curves.front()),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Svensson curves"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy floorlets"
        );

        // Warm up the specialized analytical kernel.
        const std::size_t warmup_count = std::min<std::size_t>(
            64U,
            std::min(models.size(), std::min(curves.size(), products.size()))
        );
        fitted::launch_hull_white_svensson_floorlet_cuda(
            device_models,
            warmup_count,
            device_curves,
            warmup_count,
            device_products,
            warmup_count,
            false,
            warmup_count,
            0U,
            warmup_count,
            threads_per_block,
            block_count_for(warmup_count),
            device_prices
        );
        check_cuda(cudaDeviceSynchronize(), "Hull-White floorlet warmup");

        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");

        fitted::launch_hull_white_svensson_floorlet_cuda(
            device_models,
            models.size(),
            device_curves,
            curves.size(),
            device_products,
            products.size(),
            construction == datasets::PriceConstruction::CartesianProduct,
            result_count,
            0U,
            result_count,
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

        // Copy the analytical prices back to host memory.
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy floorlet prices"
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
    check_cuda(cudaFree(device_curves), "cudaFree Svensson curves");
    check_cuda(cudaFree(device_products), "cudaFree floorlets");
    check_cuda(cudaFree(device_prices), "cudaFree floorlet prices");
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
