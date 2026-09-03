// Compare the three uniform one-block Heston product launchers on CUDA.
#include "common/check_cuda.cuh"
#include "model/equity/markovian/heston/product/asian_option.cuh"
#include "model/equity/markovian/heston/product/european_option.cuh"
#include "model/equity/markovian/heston/product/lookback_option.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr std::size_t kPathsPerPrice = 8'192U;
constexpr float kDt = 1.0f / 504.0f;
constexpr std::uint32_t kSimulationStepsPerDay = 2U;
constexpr unsigned int kThreadsPerBlock = 256U;
constexpr std::uint64_t kSeed = 900000001ULL;

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Own every device allocation shared by the three launcher checks.
struct DeviceArrays {
    ai_factory::workbench::model::equity::heston::ModelParameters* model = nullptr;
    ai_factory::workbench::product::EuropeanOptionParameters* european = nullptr;
    ai_factory::workbench::product::AsianOptionParameters* asian = nullptr;
    ai_factory::workbench::product::LookbackOptionParameters* lookback = nullptr;
    float* price = nullptr;
    float* standard_error = nullptr;

    ~DeviceArrays() {
        if (model != nullptr) cudaFree(model);
        if (european != nullptr) cudaFree(european);
        if (asian != nullptr) cudaFree(asian);
        if (lookback != nullptr) cudaFree(lookback);
        if (price != nullptr) cudaFree(price);
        if (standard_error != nullptr) cudaFree(standard_error);
    }
};

// Copy the two scalar outputs written by the most recent launcher.
void copy_outputs(
    const DeviceArrays& device,
    float& price,
    float& standard_error
) {
    using ai_factory::workbench::check_cuda;
    check_cuda(
        cudaMemcpy(
            &price, device.price, sizeof(float), cudaMemcpyDeviceToHost
        ),
        "test cudaMemcpy path-product price"
    );
    check_cuda(
        cudaMemcpy(
            &standard_error,
            device.standard_error,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "test cudaMemcpy path-product standard error"
    );
}

}  // namespace

// Verify common launch behavior and pathwise payoff ordering.
int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "path-product test cudaGetDeviceCount");

    const ai_factory::workbench::model::equity::heston::ModelParameters model = {
        1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f,
    };
    const product::EuropeanOptionParameters european = {1.0f, 252U};
    const product::AsianOptionParameters asian = {1.0f, 252U};
    const product::LookbackOptionParameters lookback = {1.0f, 252U};

    DeviceArrays device;
    check_cuda(cudaMalloc(&device.model, sizeof(model)), "test cudaMalloc model");
    check_cuda(
        cudaMalloc(&device.european, sizeof(european)),
        "test cudaMalloc European call"
    );
    check_cuda(
        cudaMalloc(&device.asian, sizeof(asian)),
        "test cudaMalloc Asian call"
    );
    check_cuda(
        cudaMalloc(&device.lookback, sizeof(lookback)),
        "test cudaMalloc lookback option"
    );
    check_cuda(cudaMalloc(&device.price, sizeof(float)), "test cudaMalloc price");
    check_cuda(
        cudaMalloc(&device.standard_error, sizeof(float)),
        "test cudaMalloc standard error"
    );
    check_cuda(
        cudaMemcpy(device.model, &model, sizeof(model), cudaMemcpyHostToDevice),
        "test cudaMemcpy model"
    );
    check_cuda(
        cudaMemcpy(
            device.european,
            &european,
            sizeof(european),
            cudaMemcpyHostToDevice
        ),
        "test cudaMemcpy European call"
    );
    check_cuda(
        cudaMemcpy(device.asian, &asian, sizeof(asian), cudaMemcpyHostToDevice),
        "test cudaMemcpy Asian call"
    );
    check_cuda(
        cudaMemcpy(
            device.lookback,
            &lookback,
            sizeof(lookback),
            cudaMemcpyHostToDevice
        ),
        "test cudaMemcpy lookback option"
    );

    float european_price = 0.0f;
    float european_error = 0.0f;
    ai_factory::workbench::model::equity::heston::launch_heston_european_option_cuda<OptionSide::call>(
        device.model, 1U, &european, device.european, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U, 1U,
        kPathsPerPrice, kDt, kSimulationStepsPerDay, kThreadsPerBlock, 1U, kSeed,
        device.price, device.standard_error
    );
    copy_outputs(device, european_price, european_error);

    float asian_price = 0.0f;
    float asian_error = 0.0f;
    ai_factory::workbench::model::equity::heston::launch_heston_asian_option_cuda<OptionSide::call>(
        device.model, 1U, &asian, device.asian, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U, 1U,
        kPathsPerPrice, kDt, kSimulationStepsPerDay, kThreadsPerBlock, 1U, kSeed,
        device.price, device.standard_error
    );
    copy_outputs(device, asian_price, asian_error);

    float lookback_price = 0.0f;
    float lookback_error = 0.0f;
    ai_factory::workbench::model::equity::heston::launch_heston_lookback_option_cuda(
        device.model, 1U, &lookback, device.lookback, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U, 1U,
        kPathsPerPrice, kDt, kSimulationStepsPerDay, kThreadsPerBlock, 1U, kSeed,
        device.price, device.standard_error
    );
    copy_outputs(device, lookback_price, lookback_error);

    require(
        std::isfinite(european_price) && std::isfinite(asian_price)
            && std::isfinite(lookback_price),
        "Heston path-product launcher returned a non-finite price"
    );
    require(
        european_error > 0.0f && asian_error > 0.0f
            && lookback_error > 0.0f,
        "Heston path-product launcher returned an invalid standard error"
    );
    require(
        lookback_price >= european_price && lookback_price >= asian_price,
        "Heston path-product payoffs violate their pathwise ordering"
    );
}
