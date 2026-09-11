// CUDA sampling checks for the prepared rough-Heston Markovian lift.
#include "common/check_cuda.cuh"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/rough_heston/sample.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace rough_heston = model::equity::rough_heston;
inline constexpr float kSampleDt = 1.0f / 504.0f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename T>
T* allocate_device(std::size_t count, const char* operation) {
    T* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, count * sizeof(T)), operation);
    return pointer;
}

template<typename T>
void copy_to_device(
    T* device_values,
    const std::vector<T>& values,
    const char* operation
) {
    check_cuda(
        cudaMemcpy(
            device_values,
            values.data(),
            values.size() * sizeof(T),
            cudaMemcpyHostToDevice
        ),
        operation
    );
}

template<typename T>
std::vector<T> copy_to_host(
    const T* device_values,
    std::size_t count,
    const char* operation
) {
    std::vector<T> values(count);
    check_cuda(
        cudaMemcpy(
            values.data(),
            device_values,
            count * sizeof(T),
            cudaMemcpyDeviceToHost
        ),
        operation
    );
    return values;
}

void validate_spots(const std::vector<float>& spots) {
    for (float spot : spots) {
        require(
            std::isfinite(spot) && spot > 0.0f,
            "A rough-Heston sample is not finite and positive."
        );
    }
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "rough-Heston sample cudaGetDeviceCount");

    constexpr std::size_t parameter_count = 4U;
    constexpr std::uint64_t seed = 923000001ULL;
    const rough_heston::ModelParameters model{
        1.0f, 0.02f, 0.01f, 0.04f, 0.30f, 0.02f,
        0.30f, 0.10f, -0.70f,
    };
    const std::vector<rough_heston::ModelParameters> models(
        parameter_count,
        model
    );
    const auto prepared = rough_heston::prepare_dynamics<2U>(
        models,
        2.0f,
        kSampleDt
    );

    auto* device_prepared = allocate_device<
        rough_heston::PreparedDynamics<2U>
    >(parameter_count, "rough-Heston prepared sample allocation");
    auto* device_first = allocate_device<float>(
        parameter_count,
        "rough-Heston first sample allocation"
    );
    auto* device_replay = allocate_device<float>(
        parameter_count,
        "rough-Heston replay sample allocation"
    );
    copy_to_device(
        device_prepared,
        prepared,
        "rough-Heston prepared sample copy"
    );

    rough_heston::launch_rough_heston_terminal_samples_cuda<2U>(
        device_prepared, parameter_count, 1U, 8U, 0U, parameter_count,
        128U, 1U, seed, device_first
    );
    rough_heston::launch_rough_heston_terminal_samples_cuda<2U>(
        device_prepared, parameter_count, 1U, 8U, 0U, parameter_count,
        256U, parameter_count, seed, device_replay
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "rough-Heston terminal sample synchronize"
    );
    const auto first = copy_to_host(
        device_first,
        parameter_count,
        "rough-Heston first sample copy"
    );
    const auto replay = copy_to_host(
        device_replay,
        parameter_count,
        "rough-Heston replay sample copy"
    );
    require(
        std::memcmp(
            first.data(),
            replay.data(),
            parameter_count * sizeof(float)
        ) == 0,
        "Rough-Heston samples changed with the launch geometry."
    );
    validate_spots(first);

    constexpr std::size_t packaged_count = 250U;
    auto* device_packaged = allocate_device<float>(
        packaged_count,
        "rough-Heston packaged sample allocation"
    );
    rough_heston::launch_rough_heston_terminal_samples_cuda<2U>(
        device_prepared, 1U, packaged_count, 4U, 0U, packaged_count,
        128U, 1U, seed + 1U, device_packaged
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "rough-Heston packaged sample synchronize"
    );
    const auto packaged = copy_to_host(
        device_packaged,
        packaged_count,
        "rough-Heston packaged sample copy"
    );
    validate_spots(packaged);
    bool has_distinct_paths = false;
    for (std::size_t path = 1U; path < packaged_count; ++path) {
        has_distinct_paths = has_distinct_paths
            || packaged[path] != packaged[0U];
    }
    require(
        has_distinct_paths,
        "The rough-Heston P=250 package repeated one path."
    );

    constexpr std::uint32_t observation_count = 3U;
    constexpr std::size_t calendar_value_count =
        observation_count * parameter_count;
    auto* device_calendar = allocate_device<float>(
        calendar_value_count,
        "rough-Heston calendar sample allocation"
    );
    rough_heston::launch_rough_heston_calendar_samples_cuda<2U>(
        device_prepared, parameter_count, 1U, 2U, 3U, observation_count,
        0U, parameter_count, 128U, parameter_count, seed + 2U,
        device_calendar
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "rough-Heston calendar sample synchronize"
    );
    validate_spots(copy_to_host(
        device_calendar,
        calendar_value_count,
        "rough-Heston calendar sample copy"
    ));

    check_cuda(cudaFree(device_calendar), "rough-Heston calendar sample free");
    check_cuda(cudaFree(device_packaged), "rough-Heston packaged sample free");
    check_cuda(cudaFree(device_replay), "rough-Heston replay sample free");
    check_cuda(cudaFree(device_first), "rough-Heston first sample free");
    check_cuda(cudaFree(device_prepared), "rough-Heston prepared sample free");
}
