// Symmetric model-only FFT sampling checks for rough Bergomi and rough SABR.
#include "common/check_cuda.cuh"
#include "model/equity/rough/rough_bergomi/sample.cuh"
#include "model/equity/rough/rough_sabr/sample.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace bergomi = model::equity::rough_bergomi;
namespace sabr = model::equity::rough_sabr;

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
            "A rough-model FFT sample is not finite and positive."
        );
    }
}

template<typename Model, typename Launcher>
void check_batch_replay(const Model* device_models, Launcher launch, std::uint64_t seed) {
    constexpr std::size_t parameters = 2U;
    constexpr std::size_t capacity = parameters * 251U;
    float* device = allocate_device<float>(capacity, "FFT split replay allocation");
    for (const std::size_t paths : {1U, 249U, 250U, 251U}) {
        const std::size_t count = parameters * paths;
        for (const std::uint32_t days : {4U, 32U, 128U, 504U}) {
            launch(device_models, parameters, paths, days, 0U, count, 1U, seed, device);
            const auto reference = copy_to_host(device, count, "FFT unsplit replay copy");
            validate_spots(reference);
            for (const std::size_t cut : {1U, 3U, 125U, 249U, 250U, 251U}) {
                if (cut >= count) continue;
                // The public output is global sample-major, including offsets.
                check_cuda(cudaMemset(device, 0xff, count*sizeof(float)), "FFT split poison");
                launch(device_models, parameters, paths, days, 0U, cut,
                       (cut + paths - 1U) / paths, seed, device);
                const auto first_slice = copy_to_host(device, count, "FFT first slice bounds copy");
                for (std::size_t index = cut; index < count; ++index) {
                    require(std::isnan(first_slice[index]), "FFT wrote beyond the first slice");
                }
                launch(device_models, parameters, paths, days, cut, count-cut,
                       parameters - cut / paths, seed, device);
                const auto replay = copy_to_host(device, count, "FFT split replay copy");
                require(std::memcmp(reference.data(), replay.data(), count*sizeof(float)) == 0,
                        "FFT samples changed at an odd batch or parameter boundary");
            }
        }
    }
    check_cuda(cudaFree(device), "FFT split replay release");
    std::cout << "FFT split replay: P=1/249/250/251, T=4/32/128/504 days, bitwise OK\n";
}

}  // namespace

int main() try {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "rough sample cudaGetDeviceCount");

    constexpr std::size_t parameter_count = 4U;
    constexpr std::uint64_t seed = 920000001ULL;
    const bergomi::ModelParameters bergomi_row{
        1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.10f, -0.70f,
    };
    const sabr::ModelParameters sabr_row{
        1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.10f, -0.70f, 1.0f,
    };
    const std::vector<bergomi::ModelParameters> bergomi_parameters(
        parameter_count,
        bergomi_row
    );
    const std::vector<sabr::ModelParameters> sabr_parameters(
        parameter_count,
        sabr_row
    );

    auto* device_bergomi = allocate_device<bergomi::ModelParameters>(
        parameter_count,
        "rough Bergomi sample parameter allocation"
    );
    auto* device_sabr = allocate_device<sabr::ModelParameters>(
        parameter_count,
        "rough SABR sample parameter allocation"
    );
    auto* device_first = allocate_device<float>(
        parameter_count,
        "rough sample first output allocation"
    );
    auto* device_replay = allocate_device<float>(
        parameter_count,
        "rough sample replay output allocation"
    );
    auto* device_sabr_spots = allocate_device<float>(
        parameter_count,
        "rough SABR sample output allocation"
    );
    copy_to_device(
        device_bergomi,
        bergomi_parameters,
        "rough Bergomi sample parameter copy"
    );
    copy_to_device(
        device_sabr,
        sabr_parameters,
        "rough SABR sample parameter copy"
    );
    check_batch_replay(device_bergomi,
        bergomi::launch_rough_bergomi_terminal_samples_cuda, seed);
    check_batch_replay(device_sabr,
        sabr::launch_rough_sabr_terminal_samples_cuda, seed);

    bergomi::launch_rough_bergomi_terminal_samples_cuda(
        device_bergomi, parameter_count, 1U, 8U, 0U, parameter_count,
        1U, seed, device_first
    );
    bergomi::launch_rough_bergomi_terminal_samples_cuda(
        device_bergomi, parameter_count, 1U, 8U, 0U, parameter_count,
        parameter_count, seed, device_replay
    );
    sabr::launch_rough_sabr_terminal_samples_cuda(
        device_sabr, parameter_count, 1U, 8U, 0U, parameter_count,
        parameter_count, seed, device_sabr_spots
    );
    check_cuda(cudaDeviceSynchronize(), "rough terminal sample synchronize");

    const auto first = copy_to_host(
        device_first,
        parameter_count,
        "rough first sample copy"
    );
    const auto replay = copy_to_host(
        device_replay,
        parameter_count,
        "rough replay sample copy"
    );
    const auto sabr_spots = copy_to_host(
        device_sabr_spots,
        parameter_count,
        "rough SABR sample copy"
    );
    require(
        std::memcmp(
            first.data(),
            replay.data(),
            parameter_count * sizeof(float)
        ) == 0,
        "Rough samples changed with the persistent block geometry."
    );
    validate_spots(first);
    validate_spots(sabr_spots);
    for (std::size_t index = 0U; index < parameter_count; ++index) {
        require(
            std::fabs(first[index] - sabr_spots[index]) < 2.0e-5f,
            "Rough SABR beta=1 does not reduce to rough Bergomi samples."
        );
    }

    constexpr std::size_t packaged_count = 250U;
    auto* device_packaged = allocate_device<float>(
        packaged_count,
        "rough packaged sample allocation"
    );
    bergomi::launch_rough_bergomi_terminal_samples_cuda(
        device_bergomi, 1U, packaged_count, 4U, 0U, packaged_count,
        1U, seed + 1U, device_packaged
    );
    check_cuda(cudaDeviceSynchronize(), "rough packaged sample synchronize");
    const auto packaged = copy_to_host(
        device_packaged,
        packaged_count,
        "rough packaged sample copy"
    );
    validate_spots(packaged);
    bool has_distinct_paths = false;
    for (std::size_t path = 1U; path < packaged_count; ++path) {
        has_distinct_paths = has_distinct_paths
            || packaged[path] != packaged[0U];
    }
    require(has_distinct_paths, "The rough P=250 package repeated one path.");

    constexpr std::uint32_t observation_count = 3U;
    constexpr std::size_t calendar_value_count =
        observation_count * parameter_count;
    auto* device_calendar = allocate_device<float>(
        calendar_value_count,
        "rough calendar sample allocation"
    );
    sabr::launch_rough_sabr_calendar_samples_cuda(
        device_sabr, parameter_count, 1U, 2U, 3U, observation_count,
        0U, parameter_count, parameter_count, seed + 2U, device_calendar
    );
    check_cuda(cudaDeviceSynchronize(), "rough calendar sample synchronize");
    const auto calendar = copy_to_host(
        device_calendar,
        calendar_value_count,
        "rough calendar sample copy"
    );
    validate_spots(calendar);

    check_cuda(cudaFree(device_calendar), "rough calendar sample free");
    check_cuda(cudaFree(device_packaged), "rough packaged sample free");
    check_cuda(cudaFree(device_sabr_spots), "rough SABR sample free");
    check_cuda(cudaFree(device_replay), "rough replay sample free");
    check_cuda(cudaFree(device_first), "rough first sample free");
    check_cuda(cudaFree(device_sabr), "rough SABR parameter free");
    check_cuda(cudaFree(device_bergomi), "rough Bergomi parameter free");
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
