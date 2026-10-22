// CUDA sampling checks for the prepared rough-Heston Markovian lift.
#include "cuda/check_cuda.cuh"
#include <sample.cuh>
#include "model/equity/rough/rough_heston/dynamics_impl.cuh"
#include "model/equity/rough/rough_heston/sample_device_preparation.cuh"
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

void check_generated_split_reference(std::size_t paths) {
    using Dynamics=rough_heston::DynamicsPolicy<7U>;
    using Policy=sample::ModelSamplingPolicy<
        simulation::FixedStepTerminalSchedule<Dynamics>,
        sample::SpotSampleObservation<Dynamics>>;
    constexpr std::size_t parameters=2U;
    const std::size_t samples=parameters*paths;
    const rough_heston::SampleParameterSampler::Configuration bounds{{
        {1.0f,1.0f},{0.01f,0.12f},{0.5f,4.0f},
        {0.01f,0.15f},{0.03f,0.25f},{-0.95f,-0.25f}}};
    auto* fused_parameters=allocate_device<rough_heston::ModelParameters>(
        parameters,"fused generated parameter allocation");
    auto* split_parameters=allocate_device<rough_heston::ModelParameters>(
        parameters,"split generated parameter allocation");
    auto* fused_days=allocate_device<std::uint32_t>(
        samples,"fused generated days allocation");
    auto* split_days=allocate_device<std::uint32_t>(
        samples,"split generated days allocation");
    auto* fused_spots=allocate_device<float>(
        samples,"fused generated spot allocation");
    auto* split_spots=allocate_device<float>(
        samples,"split generated spot allocation");
    constexpr std::uint64_t parameter_seed=4920001U;
    constexpr std::uint64_t schedule_seed=4920002U;
    constexpr std::uint64_t dynamics_seed=4920003U;
    sample::launch_prepared_samples_cuda<Policy>(
        sample::GeneratedPreparedInputSource<
            rough_heston::SampleParameterSampler,
            rough_heston::SampleDevicePreparer<7U>>{
            bounds,{2.0f,sample::kSampleFixedStepDt},fused_parameters},
        sample::UniformMaturityCalendarSource{{2U,4U},fused_days},
        {parameters,paths,0U,samples},
        sample::kFixedStepSampleTimeConfiguration,{256U,parameters},
        {parameter_seed,schedule_seed,dynamics_seed},
        {fused_spots},"rough_heston.samples",
        "generated.fused.reference","fused rough-Heston reference");
    rough_heston::launch_rough_heston_generated_random_terminal_samples_cuda<7U>(
        split_parameters,bounds,parameters,paths,2U,4U,0U,samples,
        256U,parameters,parameter_seed,schedule_seed,dynamics_seed,
        split_days,split_spots);
    check_cuda(cudaDeviceSynchronize(),"generated rough-Heston replay");
    const auto a=copy_to_host(fused_spots,samples,"fused spot copy");
    const auto b=copy_to_host(split_spots,samples,"split spot copy");
    const auto days_a=copy_to_host(fused_days,samples,"fused days copy");
    const auto days_b=copy_to_host(split_days,samples,"split days copy");
    const auto params_a=copy_to_host(fused_parameters,parameters,
        "fused parameter copy");
    const auto params_b=copy_to_host(split_parameters,parameters,
        "split parameter copy");
    require(std::memcmp(a.data(),b.data(),samples*sizeof(float))==0,
        "separate lift preparation changed generated samples");
    require(days_a==days_b,
        "separate lift preparation changed maturity days");
    require(std::memcmp(params_a.data(),params_b.data(),
        parameters*sizeof(rough_heston::ModelParameters))==0,
        "separate lift preparation changed sampled model parameters");
    check_cuda(cudaFree(fused_parameters),"fused parameter free");
    check_cuda(cudaFree(split_parameters),"split parameter free");
    check_cuda(cudaFree(fused_days),"fused days free");
    check_cuda(cudaFree(split_days),"split days free");
    check_cuda(cudaFree(fused_spots),"fused spots free");
    check_cuda(cudaFree(split_spots),"split spots free");
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

    check_generated_split_reference(1U);
    check_generated_split_reference(250U);

    check_cuda(cudaFree(device_calendar), "rough-Heston calendar sample free");
    check_cuda(cudaFree(device_packaged), "rough-Heston packaged sample free");
    check_cuda(cudaFree(device_replay), "rough-Heston replay sample free");
    check_cuda(cudaFree(device_first), "rough-Heston first sample free");
    check_cuda(cudaFree(device_prepared), "rough-Heston prepared sample free");
}
