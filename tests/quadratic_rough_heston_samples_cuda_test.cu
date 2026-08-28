// Regression for the published quadratic rough-Heston samples_02 path 77.
#include "common/check_cuda.cuh"
#include "model/equity/rough/quadratic_rough_heston/numerics.hpp"
#include "model/equity/rough/quadratic_rough_heston/sample.cuh"
#include "tools/sampling/host_philox.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {

using namespace ai_factory::workbench;
namespace quadratic = model::equity::quadratic_rough_heston;
namespace sampling = offline::sampling;

constexpr std::uint64_t kParameterSeed = 930018111ULL;
constexpr std::uint64_t kDynamicsSeed = 930018113ULL;
constexpr std::size_t kPublishedParameterIndex = 76U;
constexpr std::uint32_t kPublishedMaturityDays = 173U;
constexpr float kSampleDt = 1.0f / 504.0f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

quadratic::ModelParameters published_parameter_row() {
    sampling::HostUniformSequence uniforms(
        kParameterSeed,
        kPublishedParameterIndex
    );
    return {
        sampling::uniform({1.0f, 1.0f}, uniforms),
        sampling::uniform({0.001f, 0.08f}, uniforms),
        sampling::uniform({0.0f, 0.06f}, uniforms),
        sampling::uniform({0.03f, 0.2f}, uniforms),
        sampling::uniform({0.1f, 0.8f}, uniforms),
        sampling::uniform({0.02f, 0.2f}, uniforms),
        sampling::uniform({0.0005f, 0.02f}, uniforms),
        sampling::uniform({0.3f, 3.0f}, uniforms),
        sampling::uniform({0.3f, 2.0f}, uniforms),
        sampling::uniform({0.01f, 0.2f}, uniforms),
    };
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "quadratic rough-Heston sample cudaGetDeviceCount");

    const quadratic::ModelParameters model = published_parameter_row();
    require(
        model.feedback_rate > 2.98f
            && model.feedback_volatility > 1.95f,
        "Published quadratic rough-Heston row 77 no longer matches its seed."
    );
    const auto prepared = quadratic::prepare_dynamics<7U>(
        model,
        2.0f,
        kSampleDt
    );

    quadratic::PreparedDynamics<7U>* device_prepared = nullptr;
    float* device_first = nullptr;
    float* device_replay = nullptr;
    check_cuda(
        cudaMalloc(&device_prepared, sizeof(prepared)),
        "quadratic rough-Heston prepared allocation"
    );
    check_cuda(
        cudaMalloc(&device_first, sizeof(float)),
        "quadratic rough-Heston first output allocation"
    );
    check_cuda(
        cudaMalloc(&device_replay, sizeof(float)),
        "quadratic rough-Heston replay output allocation"
    );
    check_cuda(
        cudaMemcpy(
            device_prepared,
            &prepared,
            sizeof(prepared),
            cudaMemcpyHostToDevice
        ),
        "quadratic rough-Heston prepared copy"
    );

    // The sample engine adds the parameter index to the dynamics seed.  With
    // one local prepared row, pass the resulting published row key directly.
    constexpr std::uint64_t row_dynamics_seed =
        kDynamicsSeed + kPublishedParameterIndex;
    quadratic::launch_quadratic_rough_heston_terminal_samples_cuda<7U>(
        device_prepared,
        1U,
        1U,
        kPublishedMaturityDays,
        0U,
        1U,
        128U,
        1U,
        row_dynamics_seed,
        device_first
    );
    quadratic::launch_quadratic_rough_heston_terminal_samples_cuda<7U>(
        device_prepared,
        1U,
        1U,
        kPublishedMaturityDays,
        0U,
        1U,
        256U,
        1U,
        row_dynamics_seed,
        device_replay
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "quadratic rough-Heston row 77 synchronize"
    );

    float first = 0.0f;
    float replay = 0.0f;
    check_cuda(
        cudaMemcpy(
            &first,
            device_first,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "quadratic rough-Heston first output copy"
    );
    check_cuda(
        cudaMemcpy(
            &replay,
            device_replay,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "quadratic rough-Heston replay output copy"
    );
    require(
        std::isfinite(first) && first >= 0.0f,
        "Published quadratic rough-Heston sample row 77 is not finite."
    );
    require(
        std::memcmp(&first, &replay, sizeof(float)) == 0,
        "Quadratic rough-Heston row 77 changed with launch geometry."
    );

    check_cuda(
        cudaFree(device_replay),
        "quadratic rough-Heston replay output free"
    );
    check_cuda(
        cudaFree(device_first),
        "quadratic rough-Heston first output free"
    );
    check_cuda(
        cudaFree(device_prepared),
        "quadratic rough-Heston prepared free"
    );
}
