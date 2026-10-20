// Compile-time composition of heterogeneous rough spot policies.
#pragma once

#include "common/simulation/concepts.cuh"
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
        output.nominal_rate_dt=Head::nominal_rate_step(parameters,dt);
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
        static_assert(RateAdjustedRoughSpotPolicy<Head>,
            "Rough spot policy must declare nominal_rate_step and "
            "advance_spot_cell_with_rate for cross-asset pricing.");
        constexpr std::size_t factor=Head::kBrownianDimension==1U
            ? AssetOffset : RoughCount+MarkovAndRateFactors+SpotOffset;
        const float shock=brownian.get(factor,lane);
        if(AssetOffset%TeamSize==lane) {
            const float variance=volatility_path[
                AssetOffset*path_count*steps+slot];
            const float carry_shift =
                rate_integral_step - prepared.nominal_rate_dt;
            Head::advance_spot_cell_with_rate(
                prepared.model, variance, shock,
                log_spots[AssetOffset], carry_shift);
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


template<typename RoughPolicy, std::size_t Count>
struct HomogeneousRoughPack {
    static_assert(Count > 0U);
    static_assert(RoughPolicy::kBrownianDimension == 1U
        || RoughPolicy::kBrownianDimension == 2U);
    static constexpr std::size_t kAssetCount = Count;
    static constexpr std::size_t kFactorCount =
        Count * RoughPolicy::kBrownianDimension;
    static constexpr std::size_t kTwoFactorCount =
        RoughPolicy::kBrownianDimension == 2U ? Count : 0U;
    static constexpr bool kAllCausal =
        requires { typename RoughPolicy::VolatilityState; };

    struct Inputs {
        const typename RoughPolicy::Parameters* models;
        const std::uint32_t* scenario_indices;
    };
    struct Prepared {
        typename RoughPolicy::PreparedModel model[Count];
    };

    __device__ static void prepare(
        Prepared& output, const Inputs& input,
        std::uint32_t scenario, float dt
    ) {
        for (std::size_t asset = 0U; asset < Count; ++asset) {
            const auto index =
                input.scenario_indices[scenario * Count + asset];
            output.model[asset] =
                RoughPolicy::prepare_model(input.models[index], dt);
        }
    }

    static void append_factor_names(
        const std::vector<std::string>& external, std::size_t offset,
        std::vector<std::string>& variance_names,
        std::vector<std::string>& spot_names
    ) {
        for (std::size_t asset = 0U; asset < Count; ++asset) {
            const auto first = offset
                + asset * RoughPolicy::kBrownianDimension;
            variance_names.push_back(external[
                first + (RoughPolicy::kBrownianDimension == 2U ? 1U : 0U)]);
            if constexpr (RoughPolicy::kBrownianDimension == 2U) {
                spot_names.push_back(external[first]);
            }
        }
    }

    template<std::size_t = 0U>
    __device__ static void initial_logs(
        float* logs, float* initial, const Prepared& prepared
    ) {
        for (std::size_t asset = 0U; asset < Count; ++asset) {
            logs[asset] = prepared.model[asset].initial_log_spot;
            initial[asset] = logs[asset];
        }
    }

    template<std::size_t Dimension, std::size_t RoughCount,
             std::size_t MarkovFactors, unsigned TeamSize = 1U>
    __device__ static void advance_spots(
        const Prepared& prepared, const float* volatility_path,
        std::size_t path_count, std::uint32_t steps, std::size_t slot,
        const CorrelatedBrownianStep<Dimension, TeamSize>& brownian,
        float* log_spots, unsigned lane = 0U
    ) {
        static_assert(RoughCount == Count);
        constexpr std::size_t spot_offset =
            RoughPolicy::kBrownianDimension == 1U
                ? 0U : Count + MarkovFactors;
        for (std::size_t asset = 0U; asset < Count; ++asset) {
            const float shock = brownian.get(spot_offset + asset, lane);
            if (asset % TeamSize == lane) {
                const float variance = volatility_path[
                    asset * path_count * steps + slot];
                RoughPolicy::advance_spot_cell(
                    prepared.model[asset], variance,
                    shock, log_spots[asset]);
            }
        }
    }
};

}  // namespace ai_factory::workbench::simulation
