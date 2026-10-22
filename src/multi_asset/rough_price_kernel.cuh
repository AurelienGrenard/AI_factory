// Joint spot scan after preparing several, possibly different, rough models.
#pragma once

#include "cuda/check_cuda.cuh"
#include "philox/distributions.cuh"
#include "monte_carlo/reductions.cuh"
#include "multi_asset/markovian_pack.cuh"
#include "multi_asset/rough_pack.cuh"
#include "monte_carlo/pricing_moments_kernel.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo {

template<typename MarkovPack,typename RoughPack,typename Product>
struct HeterogeneousRoughDeviceInputs {
    typename MarkovPack::Inputs markov_models;
    typename RoughPack::Inputs rough_models;
    const typename Product::ProductParameters* products;
    const float* packed_factors;
    const RowIndexSelector* row_selectors;
};

template<typename MarkovPack,typename RoughPack,typename Product,
         bool RecordPaths=false,unsigned TeamSize=1U>
__global__ void heterogeneous_rough_price_kernel(
    HeterogeneousRoughDeviceInputs<MarkovPack,RoughPack,Product> inputs,
    std::size_t result_index,std::size_t path_offset,std::size_t path_count,
    std::uint32_t steps,std::uint32_t steps_per_day,float dt,
    std::uint64_t base_seed,const float* volatility_path,
    reductions::MomentSums* moments,float* path_payoffs,
    bool use_shared_normals,bool use_shared_lower
) {
    constexpr std::size_t rough_count=RoughPack::kAssetCount;
    constexpr std::size_t dimension=rough_count
        +MarkovPack::kFactorCount+RoughPack::kTwoFactorCount;
    constexpr std::size_t lower_count=dimension*(dimension+1U)/2U;
    __shared__ typename MarkovPack::Prepared markov_prepared;
    __shared__ typename RoughPack::Prepared rough_prepared;
    __shared__ typename Product::PreparedProduct prepared_product;
    __shared__ typename Product::ProductParameters product_parameters;
    __shared__ philox::PhiloxKey key;
    __shared__ std::uint32_t product_index;
    __shared__ std::uint32_t correlation_index;
    extern __shared__ __align__(16) unsigned char workspace[];
    constexpr std::size_t reduction_bytes=8U*sizeof(double);
    constexpr std::size_t lower_offset=reduction_bytes;
    const std::size_t state_offset=(
        lower_offset+(use_shared_lower ? lower_count*sizeof(float) : 0U)
        +alignof(typename MarkovPack::States)-1U)
        /alignof(typename MarkovPack::States)
        *alignof(typename MarkovPack::States);
    constexpr std::size_t state_count=TeamSize>1U ? 128U/TeamSize : 0U;
    const std::size_t normal_offset=(
        state_offset+state_count*sizeof(typename MarkovPack::States)
        +alignof(float)-1U)/alignof(float)*alignof(float);
    auto* team_states=reinterpret_cast<typename MarkovPack::States*>(
        workspace+state_offset);
    auto* shared_normals=reinterpret_cast<float*>(workspace+normal_offset);
    if(threadIdx.x==0U) {
        constexpr std::size_t product_slot=
            MarkovPack::kAssetCount+rough_count;
        product_index=inputs.row_selectors[product_slot].select(
            static_cast<std::uint32_t>(result_index));
        correlation_index=inputs.row_selectors[product_slot+1U].select(
            static_cast<std::uint32_t>(result_index));
        product_parameters=inputs.products[product_index];
        prepared_product=Product::prepare(product_parameters,1.0f);
        key=philox::make_key(base_seed+result_index);
    }
    __syncthreads();
    const auto* global_lower=inputs.packed_factors
        +static_cast<std::size_t>(correlation_index)*lower_count;
    auto* shared_lower=reinterpret_cast<float*>(workspace+lower_offset);
    if(use_shared_lower)
        for(std::size_t i=threadIdx.x;i<lower_count;i+=blockDim.x)
            shared_lower[i]=global_lower[i];
    __syncthreads();
    const float* lower=use_shared_lower ? shared_lower : global_lower;
    if(threadIdx.x==0U) {
        MarkovPack::template prepare_selected<0U,rough_count>(
            markov_prepared,inputs.markov_models,
            static_cast<std::uint32_t>(result_index),dt,lower);
        RoughPack::template prepare_selected<
            rough_count,MarkovPack::kFactorCount>(
            rough_prepared,inputs.rough_models,
            static_cast<std::uint32_t>(result_index),dt,lower);
    }
    __syncthreads();
    static_assert(TeamSize>=1U && TeamSize<=32U
        && (TeamSize&(TeamSize-1U))==0U);
    constexpr unsigned teams_per_block=128U/TeamSize;
    const unsigned lane=threadIdx.x%TeamSize;
    const unsigned team=threadIdx.x/TeamSize;
    double sum=0.0,sumsq=0.0;
    for(unsigned iteration=0U;iteration<TeamSize;++iteration) {
        const auto local_path=blockIdx.x*128U
            +iteration*teams_per_block+team;
        if(local_path>=path_count) continue;
        const auto path=path_offset+local_path;
        typename MarkovPack::States local_states;
        auto& markov_states=TeamSize==1U
            ? local_states : team_states[team];
        MarkovPack::template initial_states<0U,TeamSize>(
            markov_states,markov_prepared,lane);
        float rough_log_spots[rough_count];
        float rough_initial_logs[rough_count];
        RoughPack::template initial_logs<0U>(
            rough_log_spots,rough_initial_logs,rough_prepared);
        auto handler=Product::make_handler(prepared_product);
        for(std::uint32_t step=0U;step<steps;++step) {
            simulation::BrownianNormals<dimension,TeamSize> independent;
            independent.generate(key,path,step,lane);
            simulation::CorrelatedBrownianStep<dimension,TeamSize> brownian;
            if constexpr (TeamSize>1U) {
                if(use_shared_normals) {
                    auto* team_normals=shared_normals+team*dimension;
                    independent.store_to_shared(team_normals,lane);
                    brownian.calculate_from_shared(team_normals,lower,lane);
                    brownian.publish_to_shared(team_normals,lane);
                } else brownian.calculate(independent,lower,lane);
            } else brownian.calculate(independent,lower,lane);
            MarkovPack::template advance_one_step<
                0U,rough_count,dimension,TeamSize>(
                    markov_states,markov_prepared,brownian,
                    key,path,step,lane);
            const auto slot=volterra::causal_fft::history_index(
                local_path,step,steps);
            RoughPack::template advance_spots<
                dimension,rough_count,MarkovPack::kFactorCount,
                TeamSize>(
                    rough_prepared,volatility_path,path_count,
                    steps,slot,brownian,rough_log_spots,lane);
            if(Product::is_observation_step(
                step,steps,steps_per_day,product_parameters)) {
                const auto observation=
                    Product::template observe_joint_group<
                        MarkovPack,RoughPack,rough_count,TeamSize>(
                            markov_states,markov_prepared,
                            rough_log_spots,rough_initial_logs,lane);
                bool keep_going=true;
                if(lane==0U)
                    keep_going=handler.on_observation(step,observation);
                if constexpr (TeamSize>1U)
                    keep_going=__shfl_sync(
                        simulation::subgroup_mask<TeamSize>(),
                        static_cast<int>(keep_going),0U,TeamSize)!=0;
                if(!keep_going) break;
            }
        }
        if(lane==0U) {
            const float payoff=Product::finalize(prepared_product,handler);
            if constexpr (RecordPaths) path_payoffs[path]=payoff;
            sum+=static_cast<double>(payoff);
            sumsq+=static_cast<double>(payoff)*payoff;
        }
    }
    const auto sums=reductions::reduce_block(sum,sumsq);
    if(threadIdx.x==0U)
        moments[path_offset/blockDim.x+blockIdx.x]=sums;
}

template<typename MarkovPack,typename RoughPack,typename Product,
         bool RecordPaths=false,unsigned TeamSize=1U>
void launch_heterogeneous_rough_cuda(
    HeterogeneousRoughDeviceInputs<MarkovPack,RoughPack,Product> inputs,
    std::size_t result_index,std::size_t path_offset,std::size_t path_count,
    std::uint32_t steps,std::uint32_t steps_per_day,float dt,
    std::uint64_t base_seed,const float* volatility_path,
    reductions::MomentSums* moments,float* path_payoffs=nullptr,
    cudaStream_t stream=0
) {
    static_assert(RoughPack::kAssetCount>0U);
    static_assert(MarkovPack::kAssetCount+RoughPack::kAssetCount<=254U);
    if(!inputs.products || !inputs.row_selectors
        || !inputs.packed_factors
        || !volatility_path || !moments || path_count==0U
        || path_offset%128U!=0U || steps==0U || steps_per_day==0U
        || !(dt>0.0f))
        throw std::invalid_argument("invalid heterogeneous rough launch");
    if constexpr (RecordPaths) {
        if(!path_payoffs)
            throw std::invalid_argument("heterogeneous path output is null");
    }
    constexpr std::size_t dimension=RoughPack::kAssetCount
        +MarkovPack::kFactorCount+RoughPack::kTwoFactorCount;
    constexpr std::size_t lower_count=dimension*(dimension+1U)/2U;
    constexpr std::size_t lower_bytes=lower_count*sizeof(float);
    constexpr std::size_t state_count=TeamSize>1U ? 128U/TeamSize : 0U;
    constexpr std::size_t normal_bytes=TeamSize>1U
        ? dimension*(128U/TeamSize)*sizeof(float) : 0U;
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes,
        heterogeneous_rough_price_kernel<
            MarkovPack,RoughPack,Product,RecordPaths,TeamSize>),
        "heterogeneous rough kernel attributes");
    int device=0;
    check_cuda(cudaGetDevice(&device),"heterogeneous rough device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties,device),
        "heterogeneous rough device properties");
    const auto normal_offset_for=[&](bool use_shared_lower) {
        const std::size_t state_offset=(
            8U*sizeof(double)+(use_shared_lower ? lower_bytes : 0U)
            +alignof(typename MarkovPack::States)-1U)
            /alignof(typename MarkovPack::States)
            *alignof(typename MarkovPack::States);
        return (state_offset
            +state_count*sizeof(typename MarkovPack::States)
            +alignof(float)-1U)/alignof(float)*alignof(float);
    };
    const bool use_shared_lower=normal_offset_for(true)
        +attributes.sharedSizeBytes<=properties.sharedMemPerBlockOptin;
    const std::size_t normal_offset=normal_offset_for(use_shared_lower);
    const bool use_shared_normals=normal_bytes>0U
        &&normal_offset+normal_bytes+attributes.sharedSizeBytes
            <=properties.sharedMemPerBlockOptin;
    const std::size_t shared_bytes=normal_offset
        +(use_shared_normals ? normal_bytes : 0U);
    if(shared_bytes+attributes.sharedSizeBytes
        >properties.sharedMemPerBlockOptin)
        throw std::invalid_argument(
            "heterogeneous rough dimension exceeds shared memory");
    if(shared_bytes+attributes.sharedSizeBytes
        >properties.sharedMemPerBlock)
        check_cuda(cudaFuncSetAttribute(
            heterogeneous_rough_price_kernel<
                MarkovPack,RoughPack,Product,RecordPaths,TeamSize>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)),
            "heterogeneous rough shared opt-in");
    heterogeneous_rough_price_kernel<
        MarkovPack,RoughPack,Product,RecordPaths,TeamSize>
        <<<static_cast<unsigned>((path_count+127U)/128U),
           128U,shared_bytes,stream>>>(
            inputs,result_index,path_offset,path_count,steps,
            steps_per_day,dt,base_seed,volatility_path,
            moments,path_payoffs,use_shared_normals,use_shared_lower);
    check_cuda(cudaGetLastError(),"heterogeneous rough price launch");
}

}  // namespace ai_factory::workbench::monte_carlo
