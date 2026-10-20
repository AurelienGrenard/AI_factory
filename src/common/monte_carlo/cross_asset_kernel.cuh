// One correlated Brownian vector drives equities and the stochastic numeraire.
#pragma once

#include "common/check_cuda.cuh"
#include "common/multi_asset_price_row.hpp"
#include "common/philox.cuh"
#include "common/reductions.cuh"
#include "common/simulation/multi_asset_path.cuh"
#include "common/simulation/multi_rough_path.cuh"
#include "common/volterra/pricing_moments.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo {

template<typename EquityPack,typename RoughPack,typename RatePolicy,
         typename Product>
struct CrossAssetDeviceInputs {
    typename EquityPack::Inputs equity_models;
    typename RoughPack::Inputs rough_models;
    const typename RatePolicy::Parameters* rate_models;
    const std::uint32_t* rate_indices;
    const typename RatePolicy::CurveParameters* curves;
    const typename Product::ProductParameters* products;
    const MultiAssetPriceRow* rows;
    const float* packed_factors;
};

template<typename EquityPack,typename RoughPack,typename RatePolicy,
         typename Product,unsigned TeamSize>
__device__ __forceinline__ float evaluate_cross_asset_path(
    typename EquityPack::States& equity_states,
    const typename EquityPack::Prepared& equity_prepared,
    const typename RoughPack::Prepared& rough_prepared,
    const typename RatePolicy::Prepared& rate_prepared,
    const typename Product::PreparedProduct& prepared_product,
    const typename Product::ProductParameters& product_parameters,
    const float* lower,const float* volatility_path,
    float* team_innovations,std::size_t local_path,std::size_t path_count,
    philox::PhiloxKey key,std::uint64_t path,
    std::uint32_t steps,std::uint32_t steps_per_day,unsigned lane
) {
    static_assert(Product::kTerminalCashflow);
    constexpr std::size_t rough_count=RoughPack::kAssetCount;
    constexpr std::size_t rate_factor=rough_count+EquityPack::kFactorCount;
    constexpr std::size_t dimension=rate_factor+RatePolicy::kBrownianDimension
        +RoughPack::kTwoFactorCount;
    EquityPack::template initial_states<0U,TeamSize>(
        equity_states,equity_prepared,lane);
    auto rate_state=RatePolicy::initial_state(rate_prepared);
    float rough_logs[rough_count>0U ? rough_count : 1U];
    float rough_initial[rough_count>0U ? rough_count : 1U];
    if constexpr (rough_count>0U)
        RoughPack::template initial_logs<0U>(
            rough_logs,rough_initial,rough_prepared);
    auto handler=Product::make_handler(prepared_product);
    for(std::uint32_t step=0U;step<steps;++step) {
        simulation::BrownianNormals<dimension,TeamSize> independent;
        independent.generate(key,path,step,lane);
        simulation::CorrelatedBrownianStep<dimension,TeamSize> brownian;
        if constexpr (TeamSize>1U) {
            if(team_innovations) {
                independent.store_to_shared(team_innovations,lane);
                brownian.calculate_from_shared(
                    team_innovations,lower,lane);
                brownian.publish_to_shared(team_innovations,lane);
            } else brownian.calculate(independent,lower,lane);
        } else brownian.calculate(independent,lower,lane);
        float rate_brownian[RatePolicy::kBrownianDimension > 0U
                            ? RatePolicy::kBrownianDimension : 1U];
        if constexpr (RatePolicy::kBrownianDimension > 0U) {
            #pragma unroll
            for(std::size_t factor=0U;
                factor<RatePolicy::kBrownianDimension;++factor)
                rate_brownian[factor]=brownian.get(rate_factor+factor,lane);
        }
        float integrated_rate=0.0f;
        if(lane==0U)
            integrated_rate=RatePolicy::one_step_transition(
                rate_prepared,rate_brownian,key,path,step,
                static_cast<std::uint32_t>(
                    2U+EquityPack::kAssetCount+rough_count),rate_state);
        if constexpr (TeamSize>1U)
            integrated_rate=__shfl_sync(
                simulation::subgroup_mask<TeamSize>(),
                integrated_rate,0U,TeamSize);
        EquityPack::template advance_one_step_with_rate<
            0U,rough_count,dimension,TeamSize>(
                equity_states,equity_prepared,brownian,key,path,step,lane,
                integrated_rate);
        if constexpr (rough_count>0U) {
            const auto slot=volterra::causal_fft::history_index(
                local_path,step,steps);
            RoughPack::template advance_spots_with_rate<
                dimension,rough_count,
                EquityPack::kFactorCount+RatePolicy::kBrownianDimension,
                TeamSize>(
                    rough_prepared,volatility_path,path_count,steps,slot,
                    brownian,rough_logs,integrated_rate,lane);
        }
        if(Product::is_observation_step(
            step,steps,steps_per_day,product_parameters)) {
            float observation;
            if constexpr (rough_count==0U)
                observation=Product::template observe<EquityPack,TeamSize>(
                    equity_states,equity_prepared,lane);
            else observation=Product::template observe_joint_group<
                EquityPack,RoughPack,rough_count,TeamSize>(
                    equity_states,equity_prepared,
                    rough_logs,rough_initial,lane);
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
    return lane==0U ? expf(-rate_state.integral)
        *Product::finalize(prepared_product,handler) : 0.0f;
}

template<typename EquityPack,typename RoughPack,typename RatePolicy,
         typename Product,bool RecordPaths=false,unsigned TeamSize=1U,
         bool SharedStates=true>
__global__ void cross_asset_price_kernel(
    CrossAssetDeviceInputs<EquityPack,RoughPack,RatePolicy,Product> inputs,
    std::size_t result_offset,std::size_t path_offset,
    std::size_t path_count,std::size_t moment_stride,
    std::uint32_t steps_per_day,float dt,std::uint64_t base_seed,
    const float* volatility_path,volterra::PartialMoments* moments,
    float* path_payoffs,float* prices,float* standard_errors,
    bool use_shared_normals
) {
    constexpr std::size_t rough_count=RoughPack::kAssetCount;
    constexpr std::size_t dimension=rough_count+EquityPack::kFactorCount
        +RatePolicy::kBrownianDimension+RoughPack::kTwoFactorCount;
    constexpr std::size_t lower_count=dimension*(dimension+1U)/2U;
    __shared__ typename EquityPack::Prepared equity_prepared;
    __shared__ typename RoughPack::Prepared rough_prepared;
    __shared__ typename RatePolicy::Prepared rate_prepared;
    __shared__ typename Product::PreparedProduct prepared_product;
    __shared__ typename Product::ProductParameters product_parameters;
    __shared__ philox::PhiloxKey key;
    static_assert(TeamSize>=1U && TeamSize<=32U
        && (TeamSize&(TeamSize-1U))==0U);
    const unsigned teams_per_block=blockDim.x/TeamSize;
    const unsigned lane=threadIdx.x%TeamSize;
    const unsigned team=threadIdx.x/TeamSize;
    extern __shared__ double shared[];
    auto* lower=reinterpret_cast<float*>(shared);
    constexpr std::size_t lower_bytes=lower_count*sizeof(float);
    constexpr std::size_t state_offset=
        (lower_bytes+alignof(typename EquityPack::States)-1U)
        /alignof(typename EquityPack::States)
        *alignof(typename EquityPack::States);
    auto* team_states=reinterpret_cast<typename EquityPack::States*>(
        reinterpret_cast<unsigned char*>(shared)+state_offset);
    const std::size_t team_count=blockDim.x/TeamSize;
    const std::size_t normal_offset=state_offset
        +(SharedStates
            ? sizeof(typename EquityPack::States)*team_count : 0U);
    auto* team_innovations=reinterpret_cast<float*>(
        reinterpret_cast<unsigned char*>(shared)+normal_offset)
        +team*dimension;
    const std::size_t result_index=result_offset+blockIdx.y;
    if(threadIdx.x==0U) {
        const auto row=inputs.rows[result_index];
        product_parameters=inputs.products[row.product_index];
        prepared_product=Product::prepare(product_parameters,1.0f);
        EquityPack::prepare(equity_prepared,inputs.equity_models,
                            row.scenario_index,dt);
        if constexpr (rough_count>0U)
            RoughPack::prepare(rough_prepared,inputs.rough_models,
                               row.scenario_index,dt);
        typename RatePolicy::CurveParameters curve{};
        if constexpr (RatePolicy::kUsesCurve)
            curve=inputs.curves[row.curve_index];
        rate_prepared=RatePolicy::prepare(inputs.rate_models[
            inputs.rate_indices[row.scenario_index]],curve,dt);
        key=philox::make_key(base_seed+result_index);
    }
    __syncthreads();
    const auto row=inputs.rows[result_index];
    const float* global_lower=inputs.packed_factors
        +static_cast<std::size_t>(row.correlation_index)*lower_count;
    for(std::size_t index=threadIdx.x;index<lower_count;index+=blockDim.x)
        lower[index]=global_lower[index];
    __syncthreads();
    const auto steps=product_parameters.maturity_days*steps_per_day;
    double sum=0.0,sumsq=0.0;
    const std::size_t path_tiles=(path_count+blockDim.x-1U)/blockDim.x;
    for(std::size_t tile=blockIdx.x;tile<path_tiles;tile+=gridDim.x) {
        for(unsigned iteration=0U;iteration<TeamSize;++iteration) {
            const auto local_path=tile*blockDim.x
                +iteration*teams_per_block+team;
            if(local_path>=path_count) continue;
            const auto path=path_offset+local_path;
            float payoff=0.0f;
            const auto evaluate=[&](typename EquityPack::States& states) {
                return evaluate_cross_asset_path<
                    EquityPack,RoughPack,RatePolicy,Product,TeamSize>(
                        states,equity_prepared,rough_prepared,rate_prepared,
                        prepared_product,product_parameters,lower,
                        volatility_path,
                        use_shared_normals ? team_innovations : nullptr,
                        local_path,path_count,key,path,
                        steps,steps_per_day,lane);
            };
            if constexpr (TeamSize==1U || !SharedStates) {
                typename EquityPack::States states;
                payoff=evaluate(states);
            } else {
                payoff=evaluate(team_states[team]);
            }
            if constexpr (RecordPaths)
                if(lane==0U) path_payoffs[path]=payoff;
            sum+=static_cast<double>(payoff);
            sumsq+=static_cast<double>(payoff)*payoff;
        }
    }
    // All paths have finished reading L before the reduction reuses shared.
    __syncthreads();
    const auto total=reductions::reduce_block(sum,sumsq);
    if(threadIdx.x==0U) {
        if(prices) {
            double price=0.0,error=0.0;
            reductions::compute_statistics(
                total,path_count,price,error);
            prices[result_index]=static_cast<float>(price);
            standard_errors[result_index]=static_cast<float>(error);
        } else moments[blockIdx.y*moment_stride
            +path_offset/blockDim.x+blockIdx.x]=total;
    }
}

template<typename EquityPack,typename RoughPack,typename RatePolicy,
         typename Product,bool RecordPaths=false,unsigned TeamSize=1U>
void launch_cross_asset_batch_cuda(
    CrossAssetDeviceInputs<EquityPack,RoughPack,RatePolicy,Product> inputs,
    std::size_t result_offset,std::size_t result_count,
    std::size_t path_offset,std::size_t path_count,
    std::size_t moment_stride,unsigned blocks_per_result,
    unsigned threads_per_block,std::uint32_t steps_per_day,float dt,
    std::uint64_t base_seed,const float* volatility_path,
    volterra::PartialMoments* moments,float* path_payoffs=nullptr,
    float* prices=nullptr,float* standard_errors=nullptr
) {
    static_assert(EquityPack::kAssetCount+RoughPack::kAssetCount>0U);
    static_assert(EquityPack::kAssetCount+RoughPack::kAssetCount<=253U);
    static_assert(Product::kTerminalCashflow);
    static_assert(TeamSize>=1U && TeamSize<=32U
        && (TeamSize&(TeamSize-1U))==0U);
    if(!inputs.products || !inputs.rows || !inputs.packed_factors
        || !inputs.rate_models || !inputs.rate_indices
        || (RatePolicy::kUsesCurve && !inputs.curves)
        || (!moments && !prices) || (prices && !standard_errors)
        || path_count<2U
        || result_count==0U || result_count>65535U
        || blocks_per_result==0U || moment_stride<blocks_per_result
        || threads_per_block<32U || threads_per_block>1024U
        || threads_per_block%32U!=0U
        || threads_per_block%TeamSize!=0U
        || path_offset%threads_per_block!=0U
        || !(dt>0.0f) || steps_per_day==0U)
        throw std::invalid_argument("invalid cross-asset batch launch");
    if(prices && (blocks_per_result!=1U || path_offset!=0U
        || RoughPack::kAssetCount>0U))
        throw std::invalid_argument("invalid direct cross-asset result");
    if constexpr (RoughPack::kAssetCount>0U) {
        if(!volatility_path || result_count!=1U
            || threads_per_block!=128U || path_count%128U!=0U)
            throw std::invalid_argument("invalid cross-asset rough launch");
    }
    if constexpr (RecordPaths)
        if(!path_payoffs || result_count!=1U)
            throw std::invalid_argument("invalid cross-asset path output");
    constexpr std::size_t dimension=RoughPack::kAssetCount
        +EquityPack::kFactorCount+RatePolicy::kBrownianDimension
        +RoughPack::kTwoFactorCount;
    constexpr std::size_t lower_bytes=dimension*(dimension+1U)
        /2U*sizeof(float);
    constexpr std::size_t state_offset=
        (lower_bytes+alignof(typename EquityPack::States)-1U)
        /alignof(typename EquityPack::States)
        *alignof(typename EquityPack::States);
    const std::size_t teams_per_block=threads_per_block/TeamSize;
    const std::size_t state_bytes=TeamSize>1U
        ? sizeof(typename EquityPack::States)*teams_per_block : 0U;
    const std::size_t reduction_bytes=
        2U*(threads_per_block/32U)*sizeof(double);
    const std::size_t normal_bytes=TeamSize>1U
        ? dimension*sizeof(float)*teams_per_block : 0U;
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes,
        cross_asset_price_kernel<EquityPack,RoughPack,RatePolicy,
            Product,RecordPaths,TeamSize,true>),
            "cross-asset kernel attributes");
    int device=0;
    check_cuda(cudaGetDevice(&device),"cross-asset CUDA device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties,device),
        "cross-asset device properties");
    const auto fits=[&](std::size_t state,std::size_t normals) {
        const std::size_t path_bytes=state_offset+state;
        const std::size_t base=path_bytes>reduction_bytes
            ? path_bytes : reduction_bytes;
        return base+normals+attributes.sharedSizeBytes
            <=properties.sharedMemPerBlockOptin;
    };
    const bool both=TeamSize>1U
        &&fits(state_bytes,normal_bytes);
    const bool use_shared_normals=both
        ||(normal_bytes>0U &&fits(0U,normal_bytes));
    const bool use_shared_states=both
        ||(!use_shared_normals && TeamSize>1U
            &&fits(state_bytes,0U));
    const std::size_t path_bytes=state_offset
        +(use_shared_states ? state_bytes : 0U);
    const std::size_t base_shared_bytes=
        path_bytes>reduction_bytes ? path_bytes : reduction_bytes;
    const std::size_t shared_bytes=base_shared_bytes
        +(use_shared_normals ? normal_bytes : 0U);
    if(shared_bytes+attributes.sharedSizeBytes
        >properties.sharedMemPerBlockOptin)
        throw std::invalid_argument(
            "cross-asset block exceeds available shared memory");
    const dim3 grid(blocks_per_result,
        static_cast<unsigned>(result_count));
    if(use_shared_states) {
        if(shared_bytes+attributes.sharedSizeBytes
            >properties.sharedMemPerBlock)
            check_cuda(cudaFuncSetAttribute(
                cross_asset_price_kernel<EquityPack,RoughPack,RatePolicy,
                    Product,RecordPaths,TeamSize,true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_bytes)),
                "cross-asset shared memory opt-in");
        cross_asset_price_kernel<EquityPack,RoughPack,RatePolicy,
            Product,RecordPaths,TeamSize,true>
            <<<grid,threads_per_block,shared_bytes>>>(
                inputs,result_offset,path_offset,path_count,moment_stride,
                steps_per_day,dt,base_seed,volatility_path,moments,
                path_payoffs,prices,standard_errors,use_shared_normals);
    } else {
        cudaFuncAttributes local_attributes{};
        check_cuda(cudaFuncGetAttributes(&local_attributes,
            cross_asset_price_kernel<EquityPack,RoughPack,RatePolicy,
                Product,RecordPaths,TeamSize,false>),
            "cross-asset local-state kernel attributes");
        if(shared_bytes+local_attributes.sharedSizeBytes
            >properties.sharedMemPerBlockOptin)
            throw std::invalid_argument(
                "cross-asset local-state block exceeds shared memory");
        if(shared_bytes+local_attributes.sharedSizeBytes
            >properties.sharedMemPerBlock)
            check_cuda(cudaFuncSetAttribute(
                cross_asset_price_kernel<EquityPack,RoughPack,RatePolicy,
                    Product,RecordPaths,TeamSize,false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_bytes)),
                "cross-asset local-state shared memory opt-in");
        cross_asset_price_kernel<EquityPack,RoughPack,RatePolicy,
            Product,RecordPaths,TeamSize,false>
            <<<grid,threads_per_block,shared_bytes>>>(
                inputs,result_offset,path_offset,path_count,moment_stride,
                steps_per_day,dt,base_seed,volatility_path,moments,
                path_payoffs,prices,standard_errors,use_shared_normals);
    }
    check_cuda(cudaGetLastError(),"cross-asset price launch");
}

template<typename EquityPack,typename RoughPack,typename RatePolicy,
         typename Product,bool RecordPaths=false,unsigned TeamSize=1U>
void launch_cross_asset_cuda(
    CrossAssetDeviceInputs<EquityPack,RoughPack,RatePolicy,Product> inputs,
    std::size_t result_index,std::size_t path_offset,
    std::size_t path_count,std::uint32_t steps,
    std::uint32_t steps_per_day,float dt,std::uint64_t base_seed,
    const float* volatility_path,volterra::PartialMoments* moments,
    float* path_payoffs=nullptr
) {
    (void)steps;
    launch_cross_asset_batch_cuda<EquityPack,RoughPack,RatePolicy,
        Product,RecordPaths,TeamSize>(
            inputs,result_index,1U,path_offset,path_count,
            (path_count+127U)/128U,
            static_cast<unsigned>((path_count+127U)/128U),
            128U,steps_per_day,dt,base_seed,
            volatility_path,moments,path_payoffs);
}

}  // namespace ai_factory::workbench::monte_carlo
