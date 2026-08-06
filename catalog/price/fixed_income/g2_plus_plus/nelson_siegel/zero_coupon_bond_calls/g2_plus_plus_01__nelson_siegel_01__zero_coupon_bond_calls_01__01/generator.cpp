// Build one G2++ Nelson-Siegel zero-coupon bond call dataset.
#include "common/check_cuda.cuh"
#include "model/fixed_income/g2_plus_plus/nelson_siegel/zero_coupon_bond_option.cuh"
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
    "datasets/model/fixed_income/g2_plus_plus/g2_plus_plus_01.json";
const std::filesystem::path curve_dataset_path =
    "datasets/curve/nelson_siegel/nelson_siegel_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/fixed_income/zero_coupon_bond_options/zero_coupon_bond_options_01.json";

constexpr ai_factory::workbench::datasets::PriceConstruction construction =
    ai_factory::workbench::datasets::PriceConstruction::Aligned;

// CUDA configuration for the one-thread-per-price analytical kernel.
constexpr unsigned int threads_per_block = 256U;

// Artifact locations and descriptive metadata used after pricing.
const std::filesystem::path dataset_path =
    "datasets/price/fixed_income/g2_plus_plus/nelson_siegel/zero_coupon_bond_calls/"
    "g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/price/fixed_income/g2_plus_plus/nelson_siegel/zero_coupon_bond_calls/"
    "g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/price/fixed_income/g2_plus_plus/"
    "nelson_siegel/zero_coupon_bond_calls/"
    "g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01.json";
const std::string numerical_method =
    "G2++ closed-form zero-coupon bond call";

}  // namespace

// Execute the configured pricing pipeline and write all dataset artifacts.
int main() {
    using namespace ai_factory::workbench;
    namespace g2pp = model::g2_plus_plus;
    namespace fitted = g2pp::nelson_siegel;

    // 1. Load model, curve, and product rows into contiguous FP32 vectors.
    const std::vector<g2pp::G2PlusPlusModelParameters> models =
        g2pp::load_models(model_dataset_path);
    const std::vector<curve::nelson_siegel::NelsonSiegelParameters> curves =
        curve::nelson_siegel::load_curves(curve_dataset_path);
    const std::vector<product::ZeroCouponBondOptionParameters> products =
        product::load_zero_coupon_bond_options(product_dataset_path);

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
    g2pp::G2PlusPlusModelParameters* device_models = nullptr;
    curve::nelson_siegel::NelsonSiegelParameters* device_curves = nullptr;
    product::ZeroCouponBondOptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    // 3. Execute the complete GPU pipeline.
    const auto wall_start = std::chrono::system_clock::now();
    try {
        check_cuda(
            cudaMalloc(&device_models, models.size() * sizeof(models.front())),
            "cudaMalloc G2++ models"
        );
        check_cuda(
            cudaMalloc(
                &device_curves,
                curves.size() * sizeof(curves.front())
            ),
            "cudaMalloc Nelson-Siegel curves"
        );
        check_cuda(
            cudaMalloc(
                &device_products,
                products.size() * sizeof(products.front())
            ),
            "cudaMalloc zero-coupon bond calls"
        );
        check_cuda(
            cudaMalloc(&device_prices, result_count * sizeof(float)),
            "cudaMalloc zero-coupon bond call prices"
        );

        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy G2++ models"
        );
        check_cuda(
            cudaMemcpy(
                device_curves,
                curves.data(),
                curves.size() * sizeof(curves.front()),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Nelson-Siegel curves"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy zero-coupon bond calls"
        );

        // Warm up the specialized analytical kernel.
        const std::size_t warmup_count = std::min<std::size_t>(
            64U,
            std::min(models.size(), std::min(curves.size(), products.size()))
        );
        fitted::launch_g2_plus_plus_nelson_siegel_zero_coupon_bond_option_cuda<OptionSide::call>(
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
        check_cuda(cudaDeviceSynchronize(), "G2++ zero_coupon_bond_call warmup");

        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");

        fitted::launch_g2_plus_plus_nelson_siegel_zero_coupon_bond_option_cuda<OptionSide::call>(
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
            "cudaMemcpy zero-coupon bond call prices"
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
    check_cuda(cudaFree(device_models), "cudaFree G2++ models");
    check_cuda(cudaFree(device_curves), "cudaFree Nelson-Siegel curves");
    check_cuda(cudaFree(device_products), "cudaFree zero-coupon bond calls");
    check_cuda(cudaFree(device_prices), "cudaFree zero-coupon bond call prices");
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
