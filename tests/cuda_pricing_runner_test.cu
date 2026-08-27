// Exercise the offline CUDA runner independently of a catalog recipe.
#include "common/check_cuda.cuh"
#include "tools/cuda/pricing_runner.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

namespace offline_cuda = ai_factory::workbench::offline::cuda;

__global__ void analytical_kernel(
    const float* inputs,
    float* prices,
    std::size_t count
) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) prices[index] = 2.0f * inputs[index];
}

__global__ void monte_carlo_kernel(
    const float* inputs,
    float* prices,
    float* standard_errors,
    std::size_t count
) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    prices[index] = 3.0f * inputs[index];
    standard_errors[index] = 0.125f;
}

void require_close(float actual, float expected) {
    if (std::fabs(actual - expected) > 1.0e-6f) {
        throw std::runtime_error("offline CUDA runner returned wrong data");
    }
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        cudaGetLastError();
        return 77;
    }
    check_cuda(availability, "offline CUDA runner test device discovery");

    const std::vector<float> inputs{1.0f, 2.0f, 4.0f, 8.0f};
    const auto host_inputs = offline_cuda::inputs(inputs);
    constexpr unsigned int threads = 32U;

    const auto analytical_launch = [](auto& execution) {
        analytical_kernel<<<1U, threads>>>(
            execution.template input<0>(),
            execution.prices(),
            4U
        );
    };
    const offline_cuda::AnalyticalRun analytical =
        offline_cuda::run_analytical(
            host_inputs, inputs.size(), analytical_launch, analytical_launch
        );
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
        require_close(analytical.prices[index], 2.0f * inputs[index]);
    }

    const auto monte_carlo_launch = [](auto& execution) {
        monte_carlo_kernel<<<1U, threads>>>(
            execution.template input<0>(),
            execution.prices(),
            execution.standard_errors(),
            4U
        );
    };
    const offline_cuda::MonteCarloRun monte_carlo =
        offline_cuda::run_monte_carlo(
            host_inputs,
            inputs.size(),
            monte_carlo_launch,
            monte_carlo_launch
        );
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
        require_close(monte_carlo.prices[index], 3.0f * inputs[index]);
        require_close(monte_carlo.standard_errors[index], 0.125f);
    }
    if (analytical.wall_seconds < analytical.kernel_seconds
        || monte_carlo.wall_seconds < monte_carlo.kernel_seconds) {
        throw std::runtime_error("offline CUDA runner timing is inconsistent");
    }
    return 0;
}
