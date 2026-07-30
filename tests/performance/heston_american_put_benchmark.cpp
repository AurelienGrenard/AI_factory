// Benchmark 50 stratified Heston American-put rows without writing a dataset.
#include "common/check_cuda.cuh"
#include "heston/american_put.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSourceRowCount = 100U;
constexpr std::size_t kBenchmarkRowCount = 50U;
constexpr std::size_t kPathsPerPrice = 1U << 20U;
constexpr float kTargetDt = 1.0f / 252.0f;
constexpr std::uint64_t kSeed = 900000001ULL;

const std::filesystem::path model_preview_path =
    "previews/model/heston/heston_01.json";
const std::filesystem::path product_preview_path =
    "previews/product/american_puts/american_puts_01.json";

// Own the fixed device arrays used by every benchmark run.
struct DeviceArrays {
    ai_factory::workbench::heston::HestonModelParameters* models = nullptr;
    ai_factory::workbench::products::AmericanPutInput* products = nullptr;
    float* prices = nullptr;
    float* standard_errors = nullptr;

    ~DeviceArrays() {
        if (models != nullptr) cudaFree(models);
        if (products != nullptr) cudaFree(products);
        if (prices != nullptr) cudaFree(prices);
        if (standard_errors != nullptr) cudaFree(standard_errors);
    }
};

// Keep 50 evenly spaced rows so every maturity region is represented.
template <typename Value>
std::vector<Value> stratified_rows(const std::vector<Value>& source) {
    if (source.size() != kSourceRowCount) {
        throw std::runtime_error("benchmark expects exactly 100 preview rows");
    }
    std::vector<Value> selected;
    selected.reserve(kBenchmarkRowCount);
    for (std::size_t row = 0U; row < kBenchmarkRowCount; ++row) {
        selected.push_back(source[row * kSourceRowCount / kBenchmarkRowCount]);
    }
    return selected;
}

// Persist all outputs so configurations can be compared row by row.
void write_outputs(
    const std::filesystem::path& path,
    const std::vector<float>& prices,
    const std::vector<float>& standard_errors
) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot write benchmark outputs");
    }
    stream.precision(10);
    for (std::size_t row = 0U; row < prices.size(); ++row) {
        stream << row << ',' << prices[row] << ','
               << standard_errors[row] << '\n';
    }
}

}  // namespace

// Run one requested launch configuration after a short untimed warmup.
int main(int argc, char** argv) {
    using namespace ai_factory::workbench;

    if (argc != 4) {
        throw std::invalid_argument(
            "usage: benchmark <threads> <blocks_per_price> <output.csv>"
        );
    }
    const unsigned int threads_per_block =
        static_cast<unsigned int>(std::stoul(argv[1]));
    const std::size_t blocks_per_price = std::stoull(argv[2]);

    const std::vector<heston::HestonModelParameters> models =
        stratified_rows(heston::load_heston(model_preview_path));
    const std::vector<products::AmericanPutInput> products =
        stratified_rows(products::load_american_puts(product_preview_path));
    std::vector<float> prices(kBenchmarkRowCount);
    std::vector<float> standard_errors(kBenchmarkRowCount);
    DeviceArrays device;

    check_cuda(
        cudaMalloc(&device.models, models.size() * sizeof(models.front())),
        "benchmark cudaMalloc models"
    );
    check_cuda(
        cudaMalloc(&device.products, products.size() * sizeof(products.front())),
        "benchmark cudaMalloc products"
    );
    check_cuda(
        cudaMalloc(&device.prices, prices.size() * sizeof(float)),
        "benchmark cudaMalloc prices"
    );
    check_cuda(
        cudaMalloc(
            &device.standard_errors,
            standard_errors.size() * sizeof(float)
        ),
        "benchmark cudaMalloc standard errors"
    );
    check_cuda(
        cudaMemcpy(
            device.models,
            models.data(),
            models.size() * sizeof(models.front()),
            cudaMemcpyHostToDevice
        ),
        "benchmark cudaMemcpy models"
    );
    check_cuda(
        cudaMemcpy(
            device.products,
            products.data(),
            products.size() * sizeof(products.front()),
            cudaMemcpyHostToDevice
        ),
        "benchmark cudaMemcpy products"
    );

    // Warm one short row to load the module and establish normal GPU clocks.
    heston::launch_heston_american_put_cuda(
        device.models,
        1U,
        products.data(),
        device.products,
        1U,
        false,
        1U,
        4'096U,
        kTargetDt,
        threads_per_block,
        blocks_per_price,
        kSeed,
        device.prices,
        device.standard_errors
    );

    const auto wall_start = std::chrono::steady_clock::now();
    const heston::AmericanPutExecution execution =
        heston::launch_heston_american_put_cuda(
            device.models,
            models.size(),
            products.data(),
            device.products,
            products.size(),
            false,
            kBenchmarkRowCount,
            kPathsPerPrice,
            kTargetDt,
            threads_per_block,
            blocks_per_price,
            kSeed,
            device.prices,
            device.standard_errors
        );
    check_cuda(
        cudaMemcpy(
            prices.data(),
            device.prices,
            prices.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "benchmark cudaMemcpy prices"
    );
    check_cuda(
        cudaMemcpy(
            standard_errors.data(),
            device.standard_errors,
            standard_errors.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "benchmark cudaMemcpy standard errors"
    );
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();

    double price_checksum = 0.0;
    double error_checksum = 0.0;
    for (std::size_t row = 0U; row < kBenchmarkRowCount; ++row) {
        if (!std::isfinite(prices[row])
            || !std::isfinite(standard_errors[row])) {
            throw std::runtime_error("benchmark produced a non-finite output");
        }
        price_checksum += static_cast<double>(prices[row]);
        error_checksum += static_cast<double>(standard_errors[row]);
    }
    write_outputs(argv[3], prices, standard_errors);

    int device_index = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device_index), "benchmark cudaGetDevice");
    check_cuda(
        cudaGetDeviceProperties(&properties, device_index),
        "benchmark cudaGetDeviceProperties"
    );
    std::cout.precision(12);
    std::cout
        << "threads=" << threads_per_block
        << ",blocks=" << execution.blocks_per_price
        << ",sms=" << properties.multiProcessorCount
        << ",max_threads_per_sm=" << properties.maxThreadsPerMultiProcessor
        << ",registers_per_sm=" << properties.regsPerMultiprocessor
        << ",shared_bytes_per_sm=" << properties.sharedMemPerMultiprocessor
        << ",kernel_seconds=" << execution.kernel_seconds
        << ",wall_seconds=" << wall_seconds
        << ",batches=" << execution.batch_count
        << ",launches=" << execution.kernel_launch_count
        << ",max_batch=" << execution.maximum_prices_per_batch
        << ",workspace_bytes=" << execution.workspace_bytes
        << ",price_checksum=" << price_checksum
        << ",error_checksum=" << error_checksum
        << '\n';
}
