// Compile-time composition of heterogeneous rough spot policies.
#pragma once

#include "common/simulation/correlated_brownian.cuh"
#include "common/volterra/causal_fft_core.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ai_factory::workbench::simulation {

template<typename... Policies>
struct MultiRoughPack;

template<>
struct MultiRoughPack<> {
    static constexpr std::size_t kAssetCount = 0U;
    static constexpr std::size_t kFactorCount = 0U;
    static constexpr std::size_t kTwoFactorCount = 0U;
    static constexpr bool kAllCausal = true;
    struct Inputs {};
    struct Prepared {};

    __device__ static void prepare(Prepared&, const Inputs&, std::uint32_t,
                                   float) {}
    template<std::size_t>
    __device__ static void initial_logs(
        float*,float*,const Prepared&) {}
    static void append_factor_names(
        const std::vector<std::string>&,std::size_t,
        std::vector<std::string>&,std::vector<std::string>&) {}
};

template<typename Head,typename... Tail>
struct MultiRoughPack<Head,Tail...> {
    using Rest=MultiRoughPack<Tail...>;
    static constexpr std::size_t kAssetCount=1U+Rest::kAssetCount;
    static constexpr std::size_t kFactorCount=
        Head::kBrownianDimension+Rest::kFactorCount;
    static constexpr std::size_t kTwoFactorCount=
        (Head::kBrownianDimension==2U ? 1U : 0U)
            +Rest::kTwoFactorCount;
    static constexpr bool kAllCausal=
        (requires { typename Head::VolatilityState; })
            &&Rest::kAllCausal;
    static_assert(Head::kBrownianDimension==1U
        || Head::kBrownianDimension==2U);
    struct Inputs {
        const typename Head::Parameters* models;
        const std::uint32_t* scenario_indices;
        typename Rest::Inputs tail;
    };
    struct Prepared {
        typename Head::PreparedModel model;
        float nominal_rate_dt;
        typename Rest::Prepared tail;
    };

    __device__ static void prepare(
        Prepared& output,const Inputs& input,
        std::uint32_t scenario,float dt
    ) {
        const auto& parameters=input.models[input.scenario_indices[scenario]];
        output.model=Head::prepare_model(parameters,dt);
        output.nominal_rate_dt=parameters.risk_free_rate*dt;
        Rest::prepare(output.tail,input.tail,scenario,dt);
    }

    template<std::size_t AssetOffset=0U>
    __device__ static void initial_logs(
        float* logs,float* initial,const Prepared& prepared
    ) {
        logs[AssetOffset]=prepared.model.initial_log_spot;
        initial[AssetOffset]=prepared.model.initial_log_spot;
        if constexpr (Rest::kAssetCount>0U)
            Rest::template initial_logs<AssetOffset+1U>(
                logs,initial,prepared.tail);
    }

    template<std::size_t Dimension,std::size_t RoughCount,
             std::size_t MarkovFactors,unsigned TeamSize=1U,
             std::size_t AssetOffset=0U,std::size_t SpotOffset=0U>
    __device__ static void advance_spots(
        const Prepared& prepared,const float* volatility_path,
        std::size_t path_count,std::uint32_t steps,std::size_t slot,
        const CorrelatedBrownianStep<Dimension,TeamSize>& brownian,
        float* log_spots,unsigned lane=0U
    ) {
        constexpr std::size_t factor=Head::kBrownianDimension==1U
            ? AssetOffset : RoughCount+MarkovFactors+SpotOffset;
        const float shock=brownian.get(factor,lane);
        if(AssetOffset%TeamSize==lane) {
            const float variance=volatility_path[
                AssetOffset*path_count*steps+slot];
            Head::advance_spot_cell(prepared.model,variance,
                shock,log_spots[AssetOffset]);
        }
        if constexpr (Rest::kAssetCount>0U)
            Rest::template advance_spots<
                Dimension,RoughCount,MarkovFactors,TeamSize,AssetOffset+1U,
                SpotOffset+(Head::kBrownianDimension==2U ? 1U : 0U)
            >(prepared.tail,volatility_path,path_count,steps,slot,
              brownian,log_spots,lane);
    }

    template<std::size_t Dimension,std::size_t RoughCount,
             std::size_t MarkovAndRateFactors,unsigned TeamSize=1U,
             std::size_t AssetOffset=0U,std::size_t SpotOffset=0U>
    __device__ static void advance_spots_with_rate(
        const Prepared& prepared,const float* volatility_path,
        std::size_t path_count,std::uint32_t steps,std::size_t slot,
        const CorrelatedBrownianStep<Dimension,TeamSize>& brownian,
        float* log_spots,float rate_integral_step,unsigned lane=0U
    ) {
        constexpr std::size_t factor=Head::kBrownianDimension==1U
            ? AssetOffset : RoughCount+MarkovAndRateFactors+SpotOffset;
        const float shock=brownian.get(factor,lane);
        if(AssetOffset%TeamSize==lane) {
            const float variance=volatility_path[
                AssetOffset*path_count*steps+slot];
            auto adjusted=prepared.model;
            const float shift=rate_integral_step-prepared.nominal_rate_dt;
            if constexpr (requires { adjusted.drift_time_step; })
                adjusted.drift_time_step+=shift;
            else adjusted.drift_dt+=shift;
            Head::advance_spot_cell(adjusted,variance,
                shock,log_spots[AssetOffset]);
        }
        if constexpr (Rest::kAssetCount>0U)
            Rest::template advance_spots_with_rate<
                Dimension,RoughCount,MarkovAndRateFactors,TeamSize,AssetOffset+1U,
                SpotOffset+(Head::kBrownianDimension==2U ? 1U : 0U)
            >(prepared.tail,volatility_path,path_count,steps,slot,
              brownian,log_spots,rate_integral_step,lane);
    }

    static void append_factor_names(
        const std::vector<std::string>& external,
        std::size_t offset,std::vector<std::string>& variance_names,
        std::vector<std::string>& spot_names
    ) {
        variance_names.push_back(external[offset
            +(Head::kBrownianDimension==2U ? 1U : 0U)]);
        if constexpr (Head::kBrownianDimension==2U)
            spot_names.push_back(external[offset]);
        if constexpr (Rest::kAssetCount>0U)
            Rest::append_factor_names(external,
                offset+Head::kBrownianDimension,
                variance_names,spot_names);
    }
};

}  // namespace ai_factory::workbench::simulation
