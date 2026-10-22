// The FFT and path scan may be fused or separated without changing samples.
#include "cuda/check_cuda.cuh"
#include <sample.cuh>
#include "volterra/fractional_hybrid_kernel.cuh"
#include "volterra/hybrid_schedule.cuh"
#include "model/equity/rough/rough_bergomi/dynamics_impl.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
using Path=model::equity::rough_bergomi::PathPolicy;
using Policy=sample::VolterraFftModelSamplingPolicy<
    volterra::FractionalHybridKernelPolicy,Path,
    volterra::TerminalHybridSchedule,
    sample::SpotSampleObservation<Path>>;
using Parameters=model::equity::rough_bergomi::ModelParameters;
}

int main() {
    int device_count=0;
    const auto availability=cudaGetDeviceCount(&device_count);
    if(availability==cudaErrorNoDevice
        ||availability==cudaErrorInsufficientDriver
        ||device_count==0) return 77;
    check_cuda(availability,"Volterra scan parity device");
    constexpr std::size_t parameter_count=3U;
    constexpr std::size_t maximum_paths=251U;
    const std::vector<Parameters> models(parameter_count,
        {1.0f,0.0f,0.0f,0.04f,1.0f,0.12f,-0.4f});
    Parameters* device_models=nullptr;
    float* fused=nullptr;
    float* split=nullptr;
    check_cuda(cudaMalloc(&device_models,parameter_count*sizeof(Parameters)),
        "Volterra scan parity models");
    check_cuda(cudaMalloc(&fused,
        parameter_count*maximum_paths*sizeof(float)),
        "Volterra scan parity fused");
    check_cuda(cudaMalloc(&split,
        parameter_count*maximum_paths*sizeof(float)),
        "Volterra scan parity split");
    check_cuda(cudaMemcpy(device_models,models.data(),
        parameter_count*sizeof(Parameters),cudaMemcpyHostToDevice),
        "Volterra scan parity model copy");
    for(const std::size_t paths : {1U,249U,250U,251U}) {
        for(const std::uint32_t maturity : {4U,32U,128U,504U}) {
            const std::size_t count=parameter_count*paths;
            const sample::SampleRange range{
                parameter_count,paths,0U,count};
            const sample::volterra_fft::LaunchConfiguration configuration{
                2U*maturity,parameter_count};
            const sample::SamplingSeeds seeds{11U,22U,33U};
            const sample::ConstantCalendarSource<
                simulation::MaturityCalendar> calendar{{maturity}};
            const sample::DeviceParameterSource<Parameters> source{
                device_models};
            sample::volterra_fft::launch_samples_cuda<Policy>(
                source,calendar,range,configuration,seeds,{fused},
                "volterra.scan_parity","fused","Volterra fused scan",false);
            sample::volterra_fft::launch_samples_cuda<Policy>(
                source,calendar,range,configuration,seeds,{split},
                "volterra.scan_parity","split","Volterra split scan",true);
            check_cuda(cudaDeviceSynchronize(),"Volterra scan replay");
            std::vector<float> a(count),b(count);
            check_cuda(cudaMemcpy(a.data(),fused,count*sizeof(float),
                cudaMemcpyDeviceToHost),"Volterra fused copy");
            check_cuda(cudaMemcpy(b.data(),split,count*sizeof(float),
                cudaMemcpyDeviceToHost),"Volterra split copy");
            if(std::memcmp(a.data(),b.data(),count*sizeof(float))!=0)
                throw std::runtime_error(
                    "Volterra FFT scan split changed sample bits");
        }
    }
    check_cuda(cudaFree(device_models),"Volterra scan model free");
    check_cuda(cudaFree(fused),"Volterra scan fused free");
    check_cuda(cudaFree(split),"Volterra scan split free");
}
