// Exercise both compile-time sides of the shared Bates American-option engine.
#include "common/check_cuda.cuh"
#include "model/equity/markovian/bates/product/american_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using ai_factory::workbench::OptionSide;

constexpr std::size_t kRowCount = 4U;
constexpr std::size_t kPathsPerPrice = 4'096U;
constexpr unsigned int kThreadsPerBlock = 256U;
constexpr std::size_t kBlocksPerPrice = 16U;
constexpr float kDt = 1.0f / 504.0f;
constexpr std::uint32_t kSimulationStepsPerDay = 2U;
constexpr std::uint64_t kSeed = 900000001ULL;
constexpr std::size_t kExpectedKernelLaunchCount = 154U;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct DeviceArrays {
    ai_factory::workbench::model::equity::bates::ModelParameters* models = nullptr;
    ai_factory::workbench::product::AmericanOptionParameters* products = nullptr;
    float* prices = nullptr;
    float* standard_errors = nullptr;

    ~DeviceArrays() {
        if (models != nullptr) cudaFree(models);
        if (products != nullptr) cudaFree(products);
        if (prices != nullptr) cudaFree(prices);
        if (standard_errors != nullptr) cudaFree(standard_errors);
    }
};

template<OptionSide Side>
ai_factory::workbench::longstaff_schwartz::LaunchResult price_once(
    const DeviceArrays& device,
    const std::vector<
        ai_factory::workbench::product::AmericanOptionParameters
    >& products,
    std::vector<float>& prices,
    std::vector<float>& standard_errors
) {
    using namespace ai_factory::workbench;

    const longstaff_schwartz::LaunchResult execution =
        ai_factory::workbench::model::equity::bates::launch_bates_american_option_cuda<Side>(
            device.models,
            kRowCount,
            products.data(),
            device.products,
            kRowCount,
            ai_factory::workbench::PriceConstruction::Aligned,
            kRowCount,
            kPathsPerPrice,
            kDt,
            kSimulationStepsPerDay,
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
        "test cudaMemcpy American-option prices"
    );
    check_cuda(
        cudaMemcpy(
            standard_errors.data(),
            device.standard_errors,
            kRowCount * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "test cudaMemcpy American-option standard errors"
    );
    return execution;
}

template<OptionSide Side>
void validate_side(
    const DeviceArrays& device,
    const std::vector<
        ai_factory::workbench::model::equity::bates::ModelParameters
    >& models,
    const std::vector<
        ai_factory::workbench::product::AmericanOptionParameters
    >& products
) {
    using namespace ai_factory::workbench;

    std::vector<float> first_prices(kRowCount);
    std::vector<float> first_errors(kRowCount);
    std::vector<float> second_prices(kRowCount);
    std::vector<float> second_errors(kRowCount);
    const longstaff_schwartz::LaunchResult first = price_once<Side>(
        device, products, first_prices, first_errors
    );
    const longstaff_schwartz::LaunchResult second = price_once<Side>(
        device, products, second_prices, second_errors
    );
    longstaff_schwartz::validate_regression_diagnostics(
        first, "Bates American option"
    );
    longstaff_schwartz::validate_regression_diagnostics(
        second, "Bates American option"
    );

    require(first.batch_count >= 1U, "no American-option batch was launched");
    require(
        first.kernel_launch_count == kExpectedKernelLaunchCount,
        "the backward induction launched an unexpected kernel count"
    );
    require(
        first.maximum_prices_per_batch >= 1U,
        "American-option batch contains no prices"
    );
    require(
        first.blocks_per_price == kBlocksPerPrice,
        "unexpected block count per price"
    );
    require(first.workspace_bytes > 0U, "American-option workspace is empty");
    require(
        first.kernel_seconds > 0.0,
        "American-option kernel time is not positive"
    );
    require(
        first.batch_count == second.batch_count
            && first.kernel_launch_count == second.kernel_launch_count
            && first.maximum_prices_per_batch
                == second.maximum_prices_per_batch
            && first.blocks_per_price == second.blocks_per_price
            && first.workspace_bytes == second.workspace_bytes,
        "American-option execution plan is not reproducible"
    );

    for (std::size_t row = 0U; row < kRowCount; ++row) {
        const float intrinsic = [&] {
            if constexpr (Side == OptionSide::call) {
                return std::max(
                    models[row].spot - products[row].strike, 0.0f
                );
            }
            return std::max(
                products[row].strike - models[row].spot, 0.0f
            );
        }();
        require(std::isfinite(first_prices[row]), "non-finite option price");
        require(std::isfinite(first_errors[row]), "non-finite standard error");
        require(first_errors[row] >= 0.0f, "negative standard error");
        require(
            first_prices[row] + 1.0e-6f >= intrinsic,
            "American-option price is below immediate exercise"
        );
        const float upper_bound = Side == OptionSide::call
            ? models[row].spot
            : products[row].strike;
        require(
            first_prices[row] <= upper_bound + 1.0e-5f,
            "American-option price exceeds its natural upper bound"
        );
        require(
            first_prices[row] == second_prices[row]
                && first_errors[row] == second_errors[row],
            "American-option output is not exactly reproducible"
        );
    }
}

}  // namespace

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

    const std::vector<ai_factory::workbench::model::equity::bates::ModelParameters> models = {
        {1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f,
         0.20f, -0.10f, 0.15f},
        {1.0f, 0.03f, 0.00f, 0.06f, 2.0f, 0.05f, 0.40f, -0.60f,
         0.50f, -0.15f, 0.20f},
        {1.0f, 0.01f, 0.02f, 0.03f, 1.0f, 0.06f, 0.25f, -0.50f,
         0.80f, -0.05f, 0.25f},
        {1.0f, 0.04f, 0.01f, 0.08f, 2.5f, 0.07f, 0.50f, -0.40f,
         1.20f, -0.20f, 0.35f},
    };
    const std::vector<product::AmericanOptionParameters> products = {
        {0.90f, 126U, 21U},
        {1.00f, 252U, 21U},
        {1.10f, 378U, 10U},
        {1.20f, 504U, 10U},
    };

    DeviceArrays device;
    check_cuda(
        cudaMalloc(&device.models, kRowCount * sizeof(models.front())),
        "test cudaMalloc Bates models"
    );
    check_cuda(
        cudaMalloc(&device.products, kRowCount * sizeof(products.front())),
        "test cudaMalloc American options"
    );
    check_cuda(
        cudaMalloc(&device.prices, kRowCount * sizeof(float)),
        "test cudaMalloc American-option prices"
    );
    check_cuda(
        cudaMalloc(&device.standard_errors, kRowCount * sizeof(float)),
        "test cudaMalloc American-option standard errors"
    );
    check_cuda(
        cudaMemcpy(
            device.models,
            models.data(),
            kRowCount * sizeof(models.front()),
            cudaMemcpyHostToDevice
        ),
        "test cudaMemcpy Bates models"
    );
    check_cuda(
        cudaMemcpy(
            device.products,
            products.data(),
            kRowCount * sizeof(products.front()),
            cudaMemcpyHostToDevice
        ),
        "test cudaMemcpy American options"
    );

    validate_side<OptionSide::call>(device, models, products);
    validate_side<OptionSide::put>(device, models, products);
}
