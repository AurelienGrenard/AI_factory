// Product-independent Gaussian-Volterra FFT driven by correlated Brownian cells.
#pragma once

#include "common/check_cuda.cuh"
#include "common/row_index_plan.cuh"
#include "common/simulation/correlated_brownian.cuh"
#include "common/volterra/block_fft_convolution.cuh"
#include "common/volterra/causal_fft_core_kernel.cuh"
#include "common/volterra/hybrid_fft_workspace.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::volterra::hybrid_fft {

template<typename Policy>
struct CorrelatedVolatilityRow {
    using Kernel = typename Policy::KernelPolicy;
    typename Kernel::PreparedKernel kernel;
    typename Policy::PreparedModel model;
    philox::PhiloxKey key;
    const float* lower;
    float sqrt_dt;
    float dt;
};

struct CorrelatedVolatilityPlan {
    std::size_t spectrum_offset;
    std::size_t variance_offset;
    std::size_t convolution_offset;
    std::size_t bytes;
};

template<typename Policy>
CorrelatedVolatilityPlan plan_correlated_volatility_workspace(
    std::uint32_t steps,std::size_t paths
) {
    if (steps==0U || steps>kMaximumHybridFftStepCount || paths==0U
        || paths%causal_fft::kPathGroup!=0U)
        throw std::invalid_argument("invalid hybrid volatility workspace size");
    std::size_t end=sizeof(CorrelatedVolatilityRow<Policy>);
    end=(end+alignof(float2)-1U)/alignof(float2)*alignof(float2);
    const auto spectrum=end;
    std::uint32_t length=16U;
    while(length<2U*steps) length*=2U;
    end=checked_hybrid_fft_sum(end,
        checked_hybrid_fft_product(length,sizeof(float2),
            "hybrid spectrum size overflow"),
        "hybrid spectrum end overflow");
    const auto variance=end;
    end=checked_hybrid_fft_sum(end,
        checked_hybrid_fft_product(steps,sizeof(float),
            "hybrid variance size overflow"),
        "hybrid variance end overflow");
    end=(end+alignof(float2)-1U)/alignof(float2)*alignof(float2);
    const auto convolution=end;
    end=checked_hybrid_fft_sum(end,
        checked_hybrid_fft_product(
            checked_hybrid_fft_product(paths/2U,steps,
                "hybrid convolution element count overflow"),
            sizeof(float2),"hybrid convolution size overflow"),
        "hybrid convolution end overflow");
    return {spectrum,variance,convolution,end};
}

template<typename Policy,std::size_t Dimension>
__global__ void prepare_correlated_volatility_row(
    const typename Policy::Parameters* models,
    const float* packed_factors,
    std::size_t result_index,
    float dt,std::uint64_t seed,
    CorrelatedVolatilityRow<Policy>* output,
    const RowIndexSelector* row_selectors,
    std::size_t model_slot,std::size_t correlation_slot
) {
    if(threadIdx.x!=0U) return;
    const auto model_index=row_selectors[model_slot].select(
        static_cast<std::uint32_t>(result_index));
    const auto correlation_index=row_selectors[correlation_slot].select(
        static_cast<std::uint32_t>(result_index));
    const auto parameter=models[model_index];
    output->kernel=Policy::KernelPolicy::prepare(
        Policy::kernel_parameters(parameter),dt);
    output->model=Policy::prepare_model(parameter,dt);
    output->key=philox::make_key(seed+result_index);
    output->lower=packed_factors
        +static_cast<std::size_t>(correlation_index)
            *Dimension*(Dimension+1U)/2U;
    output->sqrt_dt=sqrtf(dt);
    output->dt=dt;
}

template<typename Policy,unsigned int Length,typename Forward>
__global__ void prepare_correlated_hybrid_spectrum(
    const CorrelatedVolatilityRow<Policy>* rows,
    std::uint32_t steps,float2* spectrum,float* variances
) {
    using Complex=typename Forward::value_type;
    __shared__ CorrelatedVolatilityRow<Policy> row;
    if(threadIdx.x==0U) row=*rows;
    __syncthreads();
    Complex values[Forward::storage_size];
    if(Forward::working_group::is_thread_active()) {
        #pragma unroll
        for(unsigned item=0U;item<Forward::input_ept;++item) {
            const unsigned index=item*Forward::stride+threadIdx.x;
            const float weight=index+1U<steps
                ? Policy::KernelPolicy::far_cell_weight(
                    row.kernel,index+2U) : 0.0f;
            reinterpret_cast<float2*>(values)[item]={weight,0.0f};
            if(index<steps) {
                if constexpr (Policy::kUsesVolterraVariance)
                    variances[index]=Policy::KernelPolicy::volterra_variance(
                        row.kernel,(index+1U)*row.dt);
                else variances[index]=0.0f;
            }
        }
    }
    extern __shared__ __align__(16) unsigned char scratch[];
    Forward().execute(values,reinterpret_cast<Complex*>(scratch));
    if(Forward::working_group::is_thread_active()) {
        #pragma unroll
        for(unsigned item=0U;item<Forward::output_ept;++item) {
            const unsigned index=item*Forward::stride+threadIdx.x;
            if(index<Length)
                spectrum[index]=reinterpret_cast<float2*>(values)[item];
        }
    }
}

template<typename Policy,std::size_t Dimension,unsigned int Length,
         typename Forward,typename Inverse>
__global__ void convolve_correlated_hybrid_paths(
    std::size_t path_offset,std::size_t path_count,
    std::uint32_t steps,std::size_t volatility_factor,
    const CorrelatedVolatilityRow<Policy>* rows,
    const float2* spectrum,float2* convolutions
) {
    using Complex=typename Forward::value_type;
    constexpr unsigned ffts_per_block=Forward::ffts_per_block;
    __shared__ CorrelatedVolatilityRow<Policy> row;
    if(threadIdx.x==0U && threadIdx.y==0U) row=*rows;
    __syncthreads();
    const std::size_t pair=static_cast<std::size_t>(blockIdx.x)
        *ffts_per_block+threadIdx.y;
    const std::size_t first_path=2U*pair;
    struct Loader {
        CorrelatedVolatilityRow<Policy> row;
        std::size_t path_offset,first_path,path_count,volatility_factor;
        std::uint32_t steps;
        __device__ float2 operator()(unsigned step) const {
            if(step>=steps || first_path>=path_count) return {0.0f,0.0f};
            const auto path=path_offset+first_path;
            const float first=row.sqrt_dt
                *simulation::correlated_brownian_factor(
                    row.key,path,step,volatility_factor,row.lower);
            const float second=first_path+1U<path_count
                ? row.sqrt_dt*simulation::correlated_brownian_factor(
                    row.key,path+1U,step,volatility_factor,row.lower)
                :0.0f;
            return {first,second};
        }
    };
    struct Store {
        float2* outputs;
        std::size_t pair;
        std::uint32_t steps;
        __device__ void operator()(unsigned index,float2 value) const {
            if(index<steps) outputs[pair*steps+index]=value;
        }
    };
    extern __shared__ __align__(16) unsigned char scratch[];
    execute_padded_linear_convolution<Length,Forward,Inverse>(
        spectrum,
        Loader{row,path_offset,first_path,path_count,
               volatility_factor,steps},
        Store{convolutions,pair,steps},
        reinterpret_cast<Complex*>(scratch));
}

template<typename Policy>
__global__ void reconstruct_correlated_hybrid_volatility(
    const CorrelatedVolatilityRow<Policy>* rows,
    std::size_t path_offset,std::size_t path_count,
    std::uint32_t steps,std::size_t volatility_factor,
    std::uint32_t auxiliary_source,
    const float* variances,const float2* convolutions,
    float* volatility_path
) {
    const auto path=blockIdx.x*blockDim.x+threadIdx.x;
    __shared__ CorrelatedVolatilityRow<Policy> row;
    if(threadIdx.x==0U) row=*rows;
    __syncthreads();
    if(path>=path_count) return;
    float volatility=Policy::initial_volatility(row.model);
    for(std::uint32_t step=0U;step<steps;++step) {
        volatility_path[causal_fft::history_index(path,step,steps)]
            =volatility;
        const float2 pair=step==0U ? make_float2(0.0f,0.0f)
            :convolutions[(path/2U)*steps+step-1U];
        const float far=(path&1U)!=0U ? pair.y : pair.x;
        const float rough=simulation::correlated_brownian_factor(
            row.key,path_offset+path,step,volatility_factor,row.lower);
        const auto uniforms=philox::domain_uniform_quad(
            row.key,path_offset+path,0U,
            (auxiliary_source<<24U)|step);
        const float singular=philox::box_muller(
            uniforms.first,uniforms.second).first;
        const float value=Policy::KernelPolicy::reconstruct_volterra_value(
            row.kernel,far,rough,singular);
        volatility=Policy::map_volatility(
            row.model,value,variances[step]);
    }
}

template<typename Policy,std::size_t Dimension>
void launch_correlated_hybrid_volatility_cuda(
    const typename Policy::Parameters* models,
    const float* packed_factors,std::size_t result_index,
    std::size_t path_offset,std::size_t path_count,
    float dt,std::uint32_t steps,std::uint64_t seed,
    void* workspace,std::size_t workspace_bytes,
    float* volatility_path,std::size_t volatility_factor,
    std::uint32_t auxiliary_source,
    cudaStream_t stream,
    const RowIndexSelector* row_selectors,
    std::size_t model_slot,std::size_t correlation_slot
) {
    if(!models || !row_selectors || !packed_factors
        || !workspace || !volatility_path || path_count==0U
        || path_count%causal_fft::kPathGroup!=0U || !(dt>0.0f)
        || steps==0U || steps>kMaximumHybridFftStepCount)
        throw std::invalid_argument("invalid correlated hybrid FFT launch");
    const auto plan=plan_correlated_volatility_workspace<Policy>(
        steps,path_count);
    if(workspace_bytes<plan.bytes)
        throw std::invalid_argument("correlated hybrid FFT workspace too small");
    auto* bytes=static_cast<unsigned char*>(workspace);
    auto* row=reinterpret_cast<CorrelatedVolatilityRow<Policy>*>(bytes);
    auto* spectrum=reinterpret_cast<float2*>(bytes+plan.spectrum_offset);
    auto* variances=reinterpret_cast<float*>(bytes+plan.variance_offset);
    auto* convolutions=reinterpret_cast<float2*>(
        bytes+plan.convolution_offset);
    prepare_correlated_volatility_row<Policy,Dimension><<<1U,1U,0U,stream>>>(
        models,packed_factors,result_index,dt,seed,row,
        row_selectors,model_slot,correlation_slot);
    check_cuda(cudaGetLastError(),"hybrid volatility row preparation");
    tuning::dispatch_hybrid_fft_specialization<8192U>(
        steps,[&]<typename Specialization>() {
            constexpr unsigned Length=Specialization::kLength;
            constexpr unsigned EPT=Specialization::kPricingElementsPerThread;
            constexpr unsigned FPB=Specialization::kPricingFftsPerBlock;
            using Prep=typename FftTypes<Length,EPT,1U>::Forward;
            using Forward=typename FftTypes<Length,EPT,FPB>::Forward;
            using Inverse=typename FftTypes<Length,EPT,FPB>::Inverse;
            constexpr std::size_t shared=std::max(
                static_cast<std::size_t>(Forward::shared_memory_size),
                static_cast<std::size_t>(Inverse::shared_memory_size));
            check_cuda(cudaFuncSetAttribute(
                prepare_correlated_hybrid_spectrum<Policy,Length,Prep>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(Prep::shared_memory_size)),
                "hybrid volatility spectrum shared opt-in");
            check_cuda(cudaFuncSetAttribute(
                convolve_correlated_hybrid_paths<
                    Policy,Dimension,Length,Forward,Inverse>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared)),
                "hybrid volatility convolution shared opt-in");
            prepare_correlated_hybrid_spectrum<Policy,Length,Prep>
                <<<1U,Prep::block_dim,Prep::shared_memory_size,stream>>>(
                    row,steps,spectrum,variances);
            check_cuda(cudaGetLastError(),"hybrid volatility spectrum");
            const auto pairs=path_count/2U;
            convolve_correlated_hybrid_paths<
                Policy,Dimension,Length,Forward,Inverse
            ><<<static_cast<unsigned>((pairs+FPB-1U)/FPB),
                 Forward::block_dim,shared,stream>>>(
                    path_offset,path_count,steps,volatility_factor,
                    row,spectrum,convolutions);
            check_cuda(cudaGetLastError(),"hybrid volatility convolution");
        });
    reconstruct_correlated_hybrid_volatility<Policy>
        <<<static_cast<unsigned>((path_count+127U)/128U),128U,0U,stream>>>(
            row,path_offset,path_count,steps,volatility_factor,
            auxiliary_source,variances,convolutions,volatility_path);
    check_cuda(cudaGetLastError(),"hybrid volatility reconstruction");
}

}  // namespace ai_factory::workbench::volterra::hybrid_fft
