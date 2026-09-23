// Exercise the offline CUDA runner independently of a catalog recipe.
#include "common/check_cuda.cuh"
#include "tools/cuda/pricing_runner.cuh"
#include "tools/cuda/generation_progress.hpp"
#include "tools/cuda/monte_carlo_generation_checkpoint.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
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

__global__ void monte_carlo_batch_kernel(
    const float* inputs,
    float* prices,
    float* standard_errors,
    std::size_t offset,
    std::size_t count
) {
    const std::size_t local_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_index >= count) return;
    const std::size_t index = offset + local_index;
    prices[index] = 3.0f * inputs[index];
    standard_errors[index] = 0.125f;
}

__global__ void workspace_kernel(
    const float* inputs,
    float* prices,
    float* standard_errors,
    float* workspace,
    std::size_t workspace_count
) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= workspace_count) return;
    workspace[index] = inputs[index] + 1.0f;
    prices[index] = workspace[index];
    standard_errors[index] = 0.25f;
}

void require_close(float actual, float expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::fabs(actual - expected) > 1.0e-6f) {
        throw std::runtime_error("offline CUDA runner returned wrong data");
    }
}

void check_comparison_rejects_nonfinite() {
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()}) {
        for (const bool invalid_expected : {false, true}) {
            bool rejected = false;
            try {
                require_close(invalid_expected ? 1.0f : invalid,
                              invalid_expected ? invalid : 1.0f);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("Price comparison accepted a non-finite value.");
            }
        }
    }
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;

    check_comparison_rejects_nonfinite();

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

    const std::filesystem::path progress_path =
        std::filesystem::temp_directory_path()
        / "ai_factory_generation_progress_test.json";
    setenv(
        "AI_FACTORY_GENERATION_PROGRESS",
        progress_path.c_str(),
        1
    );
    {
        offline_cuda::GenerationProgress progress(inputs.size());
        const offline_cuda::MonteCarloRun monitored =
            offline_cuda::run_monte_carlo(
                host_inputs,
                inputs.size(),
                monte_carlo_launch,
                [&](auto& execution) {
                    monte_carlo_launch(execution);
                    progress.record_cuda_progress(inputs.size());
                }
            );
        progress.complete();
        require_close(monitored.prices.back(), 3.0f * inputs.back());
    }
    unsetenv("AI_FACTORY_GENERATION_PROGRESS");
    std::ifstream progress_stream(progress_path);
    const std::string progress_document{
        std::istreambuf_iterator<char>(progress_stream),
        std::istreambuf_iterator<char>()
    };
    if (progress_document.find("\"state\": \"complete\"")
            == std::string::npos
        || progress_document.find("\"completed_prices\": 4")
            == std::string::npos) {
        throw std::runtime_error("generation progress sidecar is incomplete");
    }
    std::filesystem::remove(progress_path);
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
        require_close(monte_carlo.prices[index], 3.0f * inputs[index]);
        require_close(monte_carlo.standard_errors[index], 0.125f);
    }

    const std::filesystem::path checkpoint_directory =
        std::filesystem::temp_directory_path()
        / "ai_factory_cuda_generation_checkpoint_test";
    std::filesystem::remove_all(checkpoint_directory);
    setenv(
        "AI_FACTORY_GENERATION_CHECKPOINT_DIR",
        checkpoint_directory.c_str(),
        1
    );
    const std::string checkpoint_identity(64U, 'c');
    setenv(
        "AI_FACTORY_GENERATION_CHECKPOINT_ID",
        checkpoint_identity.c_str(),
        1
    );
    auto checkpoint_attempt = [&](std::size_t stop_after,
                                  std::size_t expected_resume,
                                  std::size_t& launch_count) {
        offline_cuda::MonteCarloGenerationCheckpoint checkpoint(inputs.size());
        if (checkpoint.completed_prices() != expected_resume) {
            throw std::runtime_error("CUDA checkpoint resumed at the wrong row");
        }
        auto result = offline_cuda::run_monte_carlo(
            host_inputs,
            inputs.size(),
            [](auto&) {},
            [&](auto& execution) {
                for (std::size_t offset = checkpoint.completed_prices();
                     offset < stop_after;) {
                    const std::size_t count = std::min<std::size_t>(
                        2U, stop_after - offset
                    );
                    checkpoint.start_batch();
                    monte_carlo_batch_kernel<<<1U, threads>>>(
                        execution.template input<0>(),
                        execution.prices(),
                        execution.standard_errors(),
                        offset,
                        count
                    );
                    ++launch_count;
                    checkpoint.commit(execution, offset, count);
                    offset += count;
                }
            }
        );
        checkpoint.restore(result);
        checkpoint.restore_kernel_seconds(result);
        return result;
    };
    std::size_t first_launch_count = 0U;
    (void) checkpoint_attempt(2U, 0U, first_launch_count);
    std::size_t resumed_launch_count = 0U;
    const auto resumed = checkpoint_attempt(
        inputs.size(), 2U, resumed_launch_count
    );
    if (first_launch_count != 1U || resumed_launch_count != 1U) {
        throw std::runtime_error("CUDA checkpoint recalculated a completed batch");
    }
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
        require_close(resumed.prices[index], 3.0f * inputs[index]);
        require_close(resumed.standard_errors[index], 0.125f);
    }
    unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_DIR");
    unsetenv("AI_FACTORY_GENERATION_CHECKPOINT_ID");
    std::filesystem::remove_all(checkpoint_directory);

    const auto workspace_launch = [](auto& execution) {
        if (execution.workspace_bytes() != 4U * sizeof(float)) {
            throw std::runtime_error(
                "offline CUDA runner allocated the wrong workspace"
            );
        }
        workspace_kernel<<<1U, threads>>>(
            execution.template input<0>(),
            execution.prices(),
            execution.standard_errors(),
            static_cast<float*>(execution.workspace()),
            4U
        );
    };
    const offline_cuda::MonteCarloRun with_workspace =
        offline_cuda::run_monte_carlo_with_workspace(
            host_inputs,
            inputs.size(),
            inputs.size() * sizeof(float),
            workspace_launch,
            workspace_launch
        );
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
        require_close(with_workspace.prices[index], inputs[index] + 1.0f);
        require_close(with_workspace.standard_errors[index], 0.25f);
    }
    if (analytical.wall_seconds < analytical.kernel_seconds
        || monte_carlo.wall_seconds < monte_carlo.kernel_seconds
        || with_workspace.wall_seconds < with_workspace.kernel_seconds) {
        throw std::runtime_error("offline CUDA runner timing is inconsistent");
    }
    return 0;
}
