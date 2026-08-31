// Regression for the published quadratic rough-Heston samples_02 path 77.
#include "common/check_cuda.cuh"
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/quadratic_rough_heston/sample.cuh"
#include "tools/sampling/host_philox.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace quadratic = model::equity::quadratic_rough_heston;
namespace sampling = offline::sampling;

// Version-1 recipe domain from capability_manifest.py.
constexpr std::uint64_t kParameterSeed = 11668828502827728896ULL;
constexpr std::uint64_t kDynamicsSeed = 11668828504975212544ULL;
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

std::vector<quadratic::ModelParameters> domain_corner_models() {
    return {
        {1.0f, 0.001f, 0.06f, 0.03f, 0.10f, 0.02f,
         0.0005f, 0.30f, 0.30f, 0.01f},
        {1.0f, 0.08f, 0.0f, 0.20f, 0.80f, 0.20f,
         0.02f, 3.0f, 2.0f, 0.20f},
        {1.0f, -0.03f, 0.10f, -0.30f, 0.02f, -0.20f,
         0.0001f, 0.05f, 0.05f, 0.005f},
        {1.0f, 0.12f, 0.0f, 0.50f, 2.0f, 0.50f,
         0.10f, 6.0f, 4.0f, 0.45f},
    };
}

template<std::size_t FactorCount>
void exercise_domain_corners() {
    constexpr std::size_t paths_per_parameter = 1'024U;
    constexpr std::uint32_t maturity_days = 504U;
    const auto models = domain_corner_models();
    const auto prepared = quadratic::prepare_dynamics<FactorCount>(
        models,
        2.0f,
        kSampleDt
    );
    const std::size_t sample_count = models.size() * paths_per_parameter;
    quadratic::PreparedDynamics<FactorCount>* device_prepared = nullptr;
    float* device_spots = nullptr;
    check_cuda(
        cudaMalloc(&device_prepared, prepared.size() * sizeof(prepared[0])),
        "quadratic rough-Heston domain prepared allocation"
    );
    check_cuda(
        cudaMalloc(&device_spots, sample_count * sizeof(float)),
        "quadratic rough-Heston domain output allocation"
    );
    check_cuda(
        cudaMemcpy(
            device_prepared,
            prepared.data(),
            prepared.size() * sizeof(prepared[0]),
            cudaMemcpyHostToDevice
        ),
        "quadratic rough-Heston domain prepared copy"
    );
    quadratic::launch_quadratic_rough_heston_terminal_samples_cuda<
        FactorCount
    >(
        device_prepared,
        models.size(),
        paths_per_parameter,
        maturity_days,
        0U,
        sample_count,
        256U,
        models.size(),
        11668828504975212544ULL,
        device_spots
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "quadratic rough-Heston domain synchronize"
    );
    std::vector<float> spots(sample_count);
    check_cuda(
        cudaMemcpy(
            spots.data(),
            device_spots,
            spots.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "quadratic rough-Heston domain output copy"
    );
    for (const float spot : spots) {
        require(
            std::isfinite(spot) && spot >= 0.0f,
            "Quadratic rough-Heston domain sweep produced a non-finite spot."
        );
    }
    check_cuda(
        cudaFree(device_spots),
        "quadratic rough-Heston domain output free"
    );
    check_cuda(
        cudaFree(device_prepared),
        "quadratic rough-Heston domain prepared free"
    );
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
        model.risk_free_rate > 0.0731f
            && model.risk_free_rate < 0.0733f
            && model.initial_feedback > 0.1534f
            && model.initial_feedback < 0.1537f,
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

    exercise_domain_corners<2U>();
    exercise_domain_corners<3U>();
    exercise_domain_corners<7U>();

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
