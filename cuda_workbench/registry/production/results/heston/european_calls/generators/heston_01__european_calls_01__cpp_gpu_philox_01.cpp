// Complete host recipe for one Heston European-call result database.
// All dataset and execution decisions are declared first; main then performs
// typed loading, CUDA pricing, timing, and canonical JSON/YAML serialization.
#include "common/check_cuda.cuh"
#include "heston/european_call.hpp"
#include "tools/registry/result_output.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace {

// Complete pricing recipe. Changing a dataset or an execution decision should
// require editing only this block.
const std::filesystem::path model_json_path =
    "cuda_workbench/registry/production/models/heston/data/heston_01.json";
const std::filesystem::path product_json_path =
    "cuda_workbench/registry/production/products/european_calls/data/european_calls_01.json";

constexpr ai_factory::workbench::ConstructionMethod construction =
    ai_factory::workbench::ConstructionMethod::Aligned;
constexpr std::size_t monte_carlo_paths_per_price = 16'384U;
constexpr float target_dt = 1.0f / 252.0f;
constexpr unsigned int threads_per_block = 512U;
constexpr std::uint64_t seed = 900000001ULL;

constexpr const char* engine = "cpp_gpu_philox";
constexpr const char* result_version = "01";
const std::filesystem::path output_root =
    "cuda_workbench/registry/production/results/heston/european_calls";
const std::filesystem::path generation_script =
    "cuda_workbench/registry/production/results/heston/european_calls/generators/"
    "heston_01__european_calls_01__cpp_gpu_philox_01.cpp";
const std::filesystem::path source_file =
    "cuda_workbench/src/heston/european_call.cu";
constexpr const char* numerical_method = "Andersen QE-M";

}  // namespace

// Execute the configured pricing pipeline from registry inputs to result files.
int main() {
    using namespace ai_factory::workbench;

    // 1. Load both JSON databases directly into contiguous FP32 vectors.
    const std::vector<heston::HestonModelInput> models =
        heston::load_heston(model_json_path);
    const std::vector<products::EuropeanCallInput> products =
        products::load_european_calls(product_json_path);

    // 2. Allocate the host outputs and declare the four device arrays.
    const std::size_t result_count = registry::constructed_row_count(
        models.size(), products.size(), construction
    );
    std::vector<float> prices(result_count);
    std::vector<float> standard_errors(result_count);
    heston::HestonModelInput* device_models = nullptr;
    products::EuropeanCallInput* device_products = nullptr;
    float* device_prices = nullptr;
    float* device_standard_errors = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    // 3. Allocate GPU memory, copy inputs, time and launch the kernel, then
    // copy both result arrays back to the CPU.
    const auto wall_start = std::chrono::steady_clock::now();
    try {
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_models),
                models.size() * sizeof(heston::HestonModelInput)
            ),
            "cudaMalloc Heston models"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_products),
                products.size() * sizeof(products::EuropeanCallInput)
            ),
            "cudaMalloc European calls"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_prices),
                result_count * sizeof(float)
            ),
            "cudaMalloc Heston call prices"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_standard_errors),
                result_count * sizeof(float)
            ),
            "cudaMalloc Heston call standard errors"
        );

        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size() * sizeof(heston::HestonModelInput),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy Heston models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(products::EuropeanCallInput),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy European calls"
        );

        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");

        heston::launch_heston_european_call_cuda(
            device_models,
            models.size(),
            device_products,
            products.size(),
            construction == ConstructionMethod::CartesianProduct,
            result_count,
            monte_carlo_paths_per_price,
            target_dt,
            threads_per_block,
            seed,
            device_prices,
            device_standard_errors
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
        kernel_seconds =
            static_cast<double>(kernel_milliseconds) * 1.0e-3;

        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy Heston call prices"
        );
        check_cuda(
            cudaMemcpy(
                standard_errors.data(),
                device_standard_errors,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy Heston call standard errors"
        );
    } catch (...) {
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
    check_cuda(cudaFree(device_models), "cudaFree Heston models");
    check_cuda(cudaFree(device_products), "cudaFree European calls");
    check_cuda(cudaFree(device_prices), "cudaFree Heston call prices");
    check_cuda(
        cudaFree(device_standard_errors),
        "cudaFree Heston call standard errors"
    );
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();

    // 5. Reconstruct and write the canonical result JSON and YAML.
    result_output::write_monte_carlo_result(
        model_json_path,
        product_json_path,
        construction,
        prices,
        standard_errors,
        engine,
        result_version,
        output_root,
        generation_script,
        source_file,
        numerical_method,
        monte_carlo_paths_per_price,
        target_dt,
        threads_per_block,
        seed,
        wall_seconds,
        kernel_seconds
    );
}
