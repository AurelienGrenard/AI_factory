// Exercise the multi-block Heston American-put pipeline on a short CUDA run.
#include "common/check_cuda.cuh"
#include "heston/american_put.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

const std::filesystem::path model_preview_path =
    "previews/model/heston/heston_01.json";
const std::filesystem::path product_preview_path =
    "previews/product/american_puts/american_puts_01.json";

constexpr std::size_t kRowCount = 4U;
constexpr std::size_t kPathsPerPrice = 4'096U;
constexpr unsigned int kThreadsPerBlock = 256U;
constexpr std::size_t kBlocksPerPrice = 16U;
constexpr float kTargetDt = 1.0f / 252.0f;
constexpr std::uint64_t kSeed = 900000001ULL;

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Own the four fixed device arrays used by this integration test.
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

// Price the same rows once and copy both FP32 outputs to the host.
ai_factory::workbench::heston::AmericanPutExecution price_once(
    const DeviceArrays& device,
    const std::vector<ai_factory::workbench::products::AmericanPutInput>& products,
    std::vector<float>& prices,
    std::vector<float>& standard_errors
) {
    using namespace ai_factory::workbench;

    const heston::AmericanPutExecution execution =
        heston::launch_heston_american_put_cuda(
            device.models,
            kRowCount,
            products.data(),
            device.products,
            kRowCount,
            false,
            kRowCount,
            kPathsPerPrice,
            kTargetDt,
            kThreadsPerBlock,
            kBlocksPerPrice,
            kSeed,
            device.prices,
            device.standard_errors
        );
    check_cuda(
        cudaMemcpy(
            prices.data(),
            device.prices,
            kRowCount * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "test cudaMemcpy American-put prices"
    );
    check_cuda(
        cudaMemcpy(
            standard_errors.data(),
            device.standard_errors,
            kRowCount * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "test cudaMemcpy American-put standard errors"
    );
    return execution;
}

}  // namespace

// Verify finite outputs, financial bounds, batching, and exact reproducibility.
int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "test cudaGetDeviceCount");

    std::vector<heston::HestonModelParameters> models =
        heston::load_heston(model_preview_path);
    std::vector<products::AmericanPutInput> products =
        products::load_american_puts(product_preview_path);
    models.resize(kRowCount);
    products.resize(kRowCount);

    DeviceArrays device;
    check_cuda(
        cudaMalloc(&device.models, kRowCount * sizeof(models.front())),
        "test cudaMalloc Heston models"
    );
    check_cuda(
        cudaMalloc(&device.products, kRowCount * sizeof(products.front())),
        "test cudaMalloc American puts"
    );
    check_cuda(
        cudaMalloc(&device.prices, kRowCount * sizeof(float)),
        "test cudaMalloc American-put prices"
    );
    check_cuda(
        cudaMalloc(&device.standard_errors, kRowCount * sizeof(float)),
        "test cudaMalloc American-put standard errors"
    );
    check_cuda(
        cudaMemcpy(
            device.models,
            models.data(),
            kRowCount * sizeof(models.front()),
            cudaMemcpyHostToDevice
        ),
        "test cudaMemcpy Heston models"
    );
    check_cuda(
        cudaMemcpy(
            device.products,
            products.data(),
            kRowCount * sizeof(products.front()),
            cudaMemcpyHostToDevice
        ),
        "test cudaMemcpy American puts"
    );

    std::vector<float> first_prices(kRowCount);
    std::vector<float> first_errors(kRowCount);
    std::vector<float> second_prices(kRowCount);
    std::vector<float> second_errors(kRowCount);
    const heston::AmericanPutExecution first = price_once(
        device, products, first_prices, first_errors
    );
    const heston::AmericanPutExecution second = price_once(
        device, products, second_prices, second_errors
    );

    require(first.batch_count >= 1U, "no American-put batch was launched");
    require(first.kernel_launch_count > 0U, "no American-put kernel was launched");
    require(
        first.maximum_prices_per_batch >= 1U,
        "American-put batch contains no prices"
    );
    require(
        first.blocks_per_price == kBlocksPerPrice,
        "unexpected block count per price"
    );
    require(first.workspace_bytes > 0U, "American-put workspace is empty");
    require(first.kernel_seconds > 0.0, "American-put kernel time is not positive");
    require(
        first.batch_count == second.batch_count
            && first.kernel_launch_count == second.kernel_launch_count
            && first.workspace_bytes == second.workspace_bytes,
        "American-put execution plan is not reproducible"
    );

    for (std::size_t row = 0U; row < kRowCount; ++row) {
        const float intrinsic = std::max(
            products[row].strike - models[row].spot, 0.0f
        );
        require(std::isfinite(first_prices[row]), "non-finite American-put price");
        require(std::isfinite(first_errors[row]), "non-finite standard error");
        require(first_errors[row] >= 0.0f, "negative standard error");
        require(
            first_prices[row] + 1.0e-6f >= intrinsic,
            "American-put price is below immediate exercise"
        );
        require(
            first_prices[row] <= products[row].strike + 1.0e-5f,
            "American-put price exceeds its strike"
        );
        require(
            first_prices[row] == second_prices[row]
                && first_errors[row] == second_errors[row],
            "American-put output is not exactly reproducible"
        );
    }
}
