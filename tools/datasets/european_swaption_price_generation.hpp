// Host pipeline shared by regular-schedule European-swaption price generators.
#pragma once

#include "common/check_cuda.cuh"
#include "product/european_swaption/dataset.hpp"
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {

inline constexpr unsigned int kEuropeanSwaptionThreadsPerBlock = 256U;
inline constexpr float kEuropeanSwaptionTimeDayFraction = 1.0f / 252.0f;

// Price one aligned model/product dataset and serialize its native CUDA output.
template<typename Model, typename Launcher>
void generate_regular_european_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const product::RegularEuropeanSwaptionDataset& product_dataset,
    Launcher launcher,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const std::string& numerical_method,
    const std::string& cuda_label
) {
    constexpr PriceConstruction construction = PriceConstruction::Aligned;
    const auto& products = product_dataset.products;
    const std::size_t result_count = price_row_count(
        models.size(), products.size(), construction
    );
    const auto block_count_for = [](std::size_t row_count) {
        return (row_count - 1U) / kEuropeanSwaptionThreadsPerBlock + 1U;
    };
    const std::size_t block_count = block_count_for(result_count);
    std::vector<float> prices(result_count);

    Model* device_models = nullptr;
    product::RegularEuropeanSwaptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    const auto wall_start = std::chrono::steady_clock::now();
    try {
        check_cuda(
            cudaMalloc(&device_models, models.size() * sizeof(models.front())),
            cuda_label + " model allocation"
        );
        check_cuda(
            cudaMalloc(
                &device_products,
                products.size() * sizeof(products.front())
            ),
            cuda_label + " product allocation"
        );
        check_cuda(
            cudaMalloc(&device_prices, result_count * sizeof(float)),
            cuda_label + " price allocation"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            cuda_label + " model copy"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            cuda_label + " product copy"
        );

        const std::size_t warmup_count = std::min<std::size_t>(
            64U, std::min(models.size(), products.size())
        );
        launcher(
            device_models,
            warmup_count,
            device_products,
            warmup_count,
            false,
            warmup_count,
            0U,
            warmup_count,
            kEuropeanSwaptionTimeDayFraction,
            kEuropeanSwaptionThreadsPerBlock,
            block_count_for(warmup_count),
            device_prices
        );
        check_cuda(cudaDeviceSynchronize(), cuda_label + " warmup");

        check_cuda(
            cudaEventCreate(&start_event), cuda_label + " start event"
        );
        check_cuda(
            cudaEventCreate(&stop_event), cuda_label + " stop event"
        );
        check_cuda(
            cudaEventRecord(start_event), cuda_label + " start record"
        );
        launcher(
            device_models,
            models.size(),
            device_products,
            products.size(),
            false,
            result_count,
            0U,
            result_count,
            kEuropeanSwaptionTimeDayFraction,
            kEuropeanSwaptionThreadsPerBlock,
            block_count,
            device_prices
        );
        check_cuda(cudaEventRecord(stop_event), cuda_label + " stop record");
        check_cuda(
            cudaEventSynchronize(stop_event), cuda_label + " stop synchronize"
        );
        float kernel_milliseconds = 0.0f;
        check_cuda(
            cudaEventElapsedTime(
                &kernel_milliseconds, start_event, stop_event
            ),
            cuda_label + " elapsed time"
        );
        kernel_seconds = static_cast<double>(kernel_milliseconds) * 1.0e-3;
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            cuda_label + " price copy"
        );
    } catch (...) {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    check_cuda(cudaEventDestroy(start_event), cuda_label + " destroy start");
    check_cuda(cudaEventDestroy(stop_event), cuda_label + " destroy stop");
    check_cuda(cudaFree(device_models), cuda_label + " free models");
    check_cuda(cudaFree(device_products), cuda_label + " free products");
    check_cuda(cudaFree(device_prices), cuda_label + " free prices");
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();

    write_analytical_price_dataset(
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
            {"threads_per_block", kEuropeanSwaptionThreadsPerBlock},
            {"kernel_launch_count", 1U},
            {"work_distribution", "one price per thread"},
        },
        wall_seconds,
        kernel_seconds
    );
    validate_price_dataset_file(dataset_path);
}

// Price one aligned model/curve/product dataset and serialize its CUDA output.
template<typename Model, typename Curve, typename Launcher>
void generate_regular_european_swaption_prices(
    const std::filesystem::path& model_dataset_path,
    const std::filesystem::path& curve_dataset_path,
    const std::filesystem::path& product_dataset_path,
    const std::vector<Model>& models,
    const std::vector<Curve>& curves,
    const product::RegularEuropeanSwaptionDataset& product_dataset,
    Launcher launcher,
    const std::filesystem::path& dataset_path,
    const std::filesystem::path& catalog_path,
    const std::string& url,
    const std::string& numerical_method,
    const std::string& cuda_label
) {
    constexpr PriceConstruction construction = PriceConstruction::Aligned;
    const auto& products = product_dataset.products;
    const std::size_t result_count = price_row_count(
        models.size(), curves.size(), products.size(), construction
    );
    const auto block_count_for = [](std::size_t row_count) {
        return (row_count - 1U) / kEuropeanSwaptionThreadsPerBlock + 1U;
    };
    const std::size_t block_count = block_count_for(result_count);
    std::vector<float> prices(result_count);

    Model* device_models = nullptr;
    Curve* device_curves = nullptr;
    product::RegularEuropeanSwaptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    const auto wall_start = std::chrono::steady_clock::now();
    try {
        check_cuda(
            cudaMalloc(&device_models, models.size() * sizeof(models.front())),
            cuda_label + " model allocation"
        );
        check_cuda(
            cudaMalloc(&device_curves, curves.size() * sizeof(curves.front())),
            cuda_label + " curve allocation"
        );
        check_cuda(
            cudaMalloc(
                &device_products,
                products.size() * sizeof(products.front())
            ),
            cuda_label + " product allocation"
        );
        check_cuda(
            cudaMalloc(&device_prices, result_count * sizeof(float)),
            cuda_label + " price allocation"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            cuda_label + " model copy"
        );
        check_cuda(
            cudaMemcpy(
                device_curves,
                curves.data(),
                curves.size() * sizeof(curves.front()),
                cudaMemcpyHostToDevice
            ),
            cuda_label + " curve copy"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            cuda_label + " product copy"
        );

        const std::size_t warmup_count = std::min<std::size_t>(
            64U,
            std::min(models.size(), std::min(curves.size(), products.size()))
        );
        launcher(
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
            kEuropeanSwaptionTimeDayFraction,
            kEuropeanSwaptionThreadsPerBlock,
            block_count_for(warmup_count),
            device_prices
        );
        check_cuda(cudaDeviceSynchronize(), cuda_label + " warmup");

        check_cuda(
            cudaEventCreate(&start_event), cuda_label + " start event"
        );
        check_cuda(
            cudaEventCreate(&stop_event), cuda_label + " stop event"
        );
        check_cuda(
            cudaEventRecord(start_event), cuda_label + " start record"
        );
        launcher(
            device_models,
            models.size(),
            device_curves,
            curves.size(),
            device_products,
            products.size(),
            false,
            result_count,
            0U,
            result_count,
            kEuropeanSwaptionTimeDayFraction,
            kEuropeanSwaptionThreadsPerBlock,
            block_count,
            device_prices
        );
        check_cuda(cudaEventRecord(stop_event), cuda_label + " stop record");
        check_cuda(
            cudaEventSynchronize(stop_event), cuda_label + " stop synchronize"
        );
        float kernel_milliseconds = 0.0f;
        check_cuda(
            cudaEventElapsedTime(
                &kernel_milliseconds, start_event, stop_event
            ),
            cuda_label + " elapsed time"
        );
        kernel_seconds = static_cast<double>(kernel_milliseconds) * 1.0e-3;
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            cuda_label + " price copy"
        );
    } catch (...) {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        if (device_models != nullptr) cudaFree(device_models);
        if (device_curves != nullptr) cudaFree(device_curves);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    check_cuda(cudaEventDestroy(start_event), cuda_label + " destroy start");
    check_cuda(cudaEventDestroy(stop_event), cuda_label + " destroy stop");
    check_cuda(cudaFree(device_models), cuda_label + " free models");
    check_cuda(cudaFree(device_curves), cuda_label + " free curves");
    check_cuda(cudaFree(device_products), cuda_label + " free products");
    check_cuda(cudaFree(device_prices), cuda_label + " free prices");
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();

    write_analytical_price_dataset(
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
            {"threads_per_block", kEuropeanSwaptionThreadsPerBlock},
            {"kernel_launch_count", 1U},
            {"work_distribution", "one price per thread"},
        },
        wall_seconds,
        kernel_seconds
    );
    validate_price_dataset_file(dataset_path);
}

}  // namespace ai_factory::workbench::datasets
