// Build one rough-Bergomi European-call price dataset from JSON inputs.
#include "common/check_cuda.cuh"
#include "model/equity/rough_bergomi/european_option.cuh"
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

const std::filesystem::path model_dataset_path =
    "datasets/model/equity/rough_bergomi/parameters/rough_bergomi_01.json";
const std::filesystem::path product_dataset_path =
    "datasets/product/equity/european_options/european_options_01.json";

constexpr ai_factory::workbench::datasets::PriceConstruction construction =
    ai_factory::workbench::datasets::PriceConstruction::Aligned;

constexpr std::size_t monte_carlo_paths_per_price = 1U << 20U;
constexpr float day_fraction = 1.0f / 252.0f;
constexpr float target_dt = 1.0f / 360.0f;
constexpr unsigned int threads_per_block = 512U;
constexpr std::size_t results_per_kernel_launch = 1'000U;
constexpr std::size_t block_count = 128U;
constexpr std::uint64_t seed = 910000001ULL;

const std::string target_dt_description = "1 / 360";
const std::string random_generator = "Philox";
const std::filesystem::path dataset_path =
    "datasets/model/equity/rough_bergomi/prices/european_calls/"
    "rough_bergomi_01__european_calls_01__01.json";
const std::filesystem::path catalog_path =
    "catalog/model/equity/rough_bergomi/prices/european_calls/"
    "rough_bergomi_01__european_calls_01__01/dataset.yaml";
const std::string url =
    "https://datasets.ai-factory.example/v1/model/equity/rough_bergomi/prices/"
    "european_calls/rough_bergomi_01__european_calls_01__01.json";
const std::string numerical_method =
    "Bennedsen-Lunde-Pakkanen hybrid scheme (kappa=1)";

}  // namespace

int main() {
    using namespace ai_factory::workbench;

    const std::vector<rough_bergomi::RoughBergomiModelParameters> models =
        rough_bergomi::load_models(model_dataset_path);
    const std::vector<product::EuropeanOptionParameters> products =
        product::load_european_options(product_dataset_path);
    const std::size_t result_count = datasets::price_row_count(
        models.size(), products.size(), construction
    );
    const std::size_t kernel_launch_count =
        (result_count + results_per_kernel_launch - 1U)
        / results_per_kernel_launch;
    const std::size_t maximum_block_count = bounded_block_count(
        std::min(result_count, results_per_kernel_launch), block_count
    );
    const rough_bergomi::RoughBergomiWorkspacePlan workspace =
        rough_bergomi::plan_european_option_workspace(
            products.data(),
            products.size(),
            day_fraction,
            target_dt,
            threads_per_block,
            maximum_block_count
        );

    std::vector<float> prices(result_count);
    std::vector<float> standard_errors(result_count);

    rough_bergomi::RoughBergomiModelParameters* device_models = nullptr;
    product::EuropeanOptionParameters* device_products = nullptr;
    float* device_history_workspace = nullptr;
    float* device_prices = nullptr;
    float* device_standard_errors = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;

    const auto wall_start = std::chrono::steady_clock::now();
    try {
        check_cuda(
            cudaMalloc(
                &device_models,
                models.size()
                    * sizeof(rough_bergomi::RoughBergomiModelParameters)
            ),
            "cudaMalloc rough-Bergomi models"
        );
        check_cuda(
            cudaMalloc(
                &device_products,
                products.size() * sizeof(product::EuropeanOptionParameters)
            ),
            "cudaMalloc European calls"
        );
        check_cuda(
            cudaMalloc(&device_history_workspace, workspace.history_bytes),
            "cudaMalloc rough-Bergomi call history"
        );
        check_cuda(
            cudaMalloc(&device_prices, result_count * sizeof(float)),
            "cudaMalloc rough-Bergomi call prices"
        );
        check_cuda(
            cudaMalloc(
                &device_standard_errors, result_count * sizeof(float)
            ),
            "cudaMalloc rough-Bergomi call standard errors"
        );

        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                models.size()
                    * sizeof(rough_bergomi::RoughBergomiModelParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy rough-Bergomi models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                products.size() * sizeof(product::EuropeanOptionParameters),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy European calls"
        );

        const std::size_t warmup_row_count = std::min<std::size_t>(
            64U, std::min(models.size(), products.size())
        );
        const std::size_t warmup_block_count = bounded_block_count(
            warmup_row_count, block_count
        );
        rough_bergomi::launch_rough_bergomi_european_option_cuda<
            OptionSide::call
        >(
            device_models,
            warmup_row_count,
            device_products,
            warmup_row_count,
            false,
            warmup_row_count,
            0U,
            warmup_row_count,
            monte_carlo_paths_per_price,
            day_fraction,
            target_dt,
            threads_per_block,
            warmup_block_count,
            workspace.maximum_step_count,
            device_history_workspace,
            workspace.history_float_count,
            seed,
            device_prices,
            device_standard_errors
        );
        check_cuda(cudaDeviceSynchronize(), "rough-Bergomi call warmup");

        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");

        for (std::size_t result_offset = 0U;
             result_offset < result_count;
             result_offset += results_per_kernel_launch) {
            const std::size_t launch_result_count = std::min(
                results_per_kernel_launch, result_count - result_offset
            );
            const std::size_t launched_block_count = bounded_block_count(
                launch_result_count, block_count
            );
            rough_bergomi::launch_rough_bergomi_european_option_cuda<
                OptionSide::call
            >(
                device_models,
                models.size(),
                device_products,
                products.size(),
                construction
                    == datasets::PriceConstruction::CartesianProduct,
                result_count,
                result_offset,
                launch_result_count,
                monte_carlo_paths_per_price,
                day_fraction,
                target_dt,
                threads_per_block,
                launched_block_count,
                workspace.maximum_step_count,
                device_history_workspace,
                workspace.history_float_count,
                seed,
                device_prices,
                device_standard_errors
            );
        }

        check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
        check_cuda(
            cudaEventSynchronize(stop_event), "cudaEventSynchronize stop"
        );
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
            "cudaMemcpy rough-Bergomi call prices"
        );
        check_cuda(
            cudaMemcpy(
                standard_errors.data(),
                device_standard_errors,
                result_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy rough-Bergomi call standard errors"
        );
    } catch (...) {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_history_workspace != nullptr)
            cudaFree(device_history_workspace);
        if (device_prices != nullptr) cudaFree(device_prices);
        if (device_standard_errors != nullptr)
            cudaFree(device_standard_errors);
        throw;
    }

    check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
    check_cuda(cudaFree(device_models), "cudaFree rough-Bergomi models");
    check_cuda(cudaFree(device_products), "cudaFree European calls");
    check_cuda(
        cudaFree(device_history_workspace),
        "cudaFree rough-Bergomi call history"
    );
    check_cuda(cudaFree(device_prices), "cudaFree rough-Bergomi call prices");
    check_cuda(
        cudaFree(device_standard_errors),
        "cudaFree rough-Bergomi call standard errors"
    );
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();

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
            {
                "maximum_prices_per_block",
                (std::min(result_count, results_per_kernel_launch)
                    + maximum_block_count - 1U)
                    / maximum_block_count
            },
            {"maximum_step_count", workspace.maximum_step_count},
            {"history_workspace_bytes", workspace.history_bytes},
            {"dynamic_shared_bytes", workspace.dynamic_shared_bytes},
            {"history_layout", "block, step, thread"},
        },
        nlohmann::ordered_json::object(),
        seed,
        wall_seconds,
        kernel_seconds
    );
    datasets::validate_price_dataset_file(dataset_path);
}
