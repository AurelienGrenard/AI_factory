// Static composition of independent model states driven by one Brownian vector.
#pragma once

#include "common/simulation/concepts.cuh"
#include "common/row_index_plan.cuh"
#include "common/simulation/correlated_brownian.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::simulation {

template<typename DynamicsPolicy, std::size_t Count>
struct ModelGroup {
    static_assert(Count > 0U);
    static_assert(CorrelatedStepDynamicsPolicy<
        DynamicsPolicy, philox::DomainUniformSequence>);
    using Dynamics = DynamicsPolicy;
    using Parameters = typename Dynamics::Parameters;
    using PreparedDynamics = typename Dynamics::PreparedDynamics;
    using State = typename Dynamics::State;
    static constexpr std::size_t kAssetCount = Count;
    static constexpr std::size_t kFactorCount =
        Count * Dynamics::kBrownianDimension;
    static_assert(Dynamics::kBrownianDimension == 1U
        || Dynamics::kBrownianDimension == 2U);

    struct Inputs {
        const Parameters* models;
        const RowIndexSelector* row_selectors; // [Count]
    };
    struct Prepared {
        PreparedDynamics model[Count];
        float initial_spot[Count];
    };
    struct States {
        State model[Count];
    };

    template<std::size_t FactorOffset>
    __device__ __forceinline__ static void prepare_selected(
        Prepared& output, const Inputs& input, std::uint32_t row,
        float dt, const float* packed_lower
    ) {
        for (std::size_t i = 0U; i < Count; ++i) {
            auto model = input.models[input.row_selectors[i].select(row)];
            if constexpr (requires { model.risk_free_rate; })
                model.risk_free_rate = 0.0f;
            if constexpr (Dynamics::kBrownianDimension == 2U) {
                const float rho = factor_correlation(
                    packed_lower, FactorOffset + 2U * i,
                    FactorOffset + 2U * i + 1U);
                if constexpr (requires { model.rho; }) model.rho = rho;
                else if constexpr (requires { model.correlation; })
                    model.correlation = rho;
                else static_assert(Dynamics::kBrownianDimension != 2U,
                                   "Two-factor model needs rho.");
            }
            output.model[i] = Dynamics::prepare_dynamics(model, dt);
            output.initial_spot[i] =
                Dynamics::spot(Dynamics::initial_state(output.model[i]));
        }
    }

    template<std::size_t AssetOffset, unsigned int TeamSize>
    __device__ __forceinline__ static void initial_states(
        States& states, const Prepared& prepared, unsigned int lane
    ) {
        constexpr std::size_t lane_shift=AssetOffset%TeamSize;
        constexpr std::size_t local_count=(Count+TeamSize-1U)/TeamSize;
        for(std::size_t local=0U;local<local_count;++local) {
            const std::size_t i=local*TeamSize
                +(lane+TeamSize-lane_shift)%TeamSize;
            if(i<Count)
                states.model[i]=Dynamics::initial_state(prepared.model[i]);
        }
    }

    template<
        std::size_t AssetOffset, std::size_t FactorOffset,
        std::size_t Dimension, unsigned int TeamSize,
        typename Transition
    >
    __device__ __forceinline__ static void advance_one_step_with_transition(
        States& states,
        const Prepared& prepared,
        const CorrelatedBrownianStep<Dimension, TeamSize>& brownian_vector,
        philox::PhiloxKey key,
        std::uint64_t path,
        std::uint32_t step,
        unsigned int lane,
        Transition&& apply_transition
    ) {
        static_assert(AssetOffset + Count + 1U <= 255U,
            "The multi-asset Philox domain supports at most 254 model slots.");
        const auto innovation=[&](std::size_t i) {
            const std::size_t first=
                FactorOffset+i*Dynamics::kBrownianDimension;
            const float spot=brownian_vector.get(first,lane);
            if constexpr (Dynamics::kBrownianDimension==1U)
                return typename Dynamics::BrownianStep{spot};
            else return typename Dynamics::BrownianStep{
                spot,brownian_vector.get(first+1U,lane)};
        };
        const auto transition=[&](
            std::size_t i,const typename Dynamics::BrownianStep& shock
        ) {
            const auto source=static_cast<std::uint32_t>(
                2U+AssetOffset+i);
            philox::DomainUniformSequence auxiliary(
                key,path,(source<<24U)|step);
            apply_transition(
                i, prepared.model[i], shock, auxiliary, states.model[i]);
        };
        bool distributed=TeamSize==1U;
        if constexpr (TeamSize>1U)
            distributed=brownian_vector.has_shared_factors();
        if(distributed) {
            constexpr std::size_t lane_shift=AssetOffset%TeamSize;
            constexpr std::size_t local_count=
                (Count+TeamSize-1U)/TeamSize;
            for(std::size_t local=0U;local<local_count;++local) {
                const std::size_t i=local*TeamSize
                    +(lane+TeamSize-lane_shift)%TeamSize;
                const auto shock=innovation(i<Count ? i : 0U);
                if(i<Count) transition(i,shock);
            }
        } else {
            for(std::size_t i=0U;i<Count;++i) {
                const auto shock=innovation(i);
                if((AssetOffset+i)%TeamSize==lane)
                    transition(i,shock);
                if constexpr (TeamSize>1U)
                    __syncwarp(subgroup_mask<TeamSize>());
            }
        }
    }

    template<
        std::size_t AssetOffset, std::size_t FactorOffset,
        std::size_t Dimension, unsigned int TeamSize
    >
    __device__ __forceinline__ static void advance_one_step(
        States& states, const Prepared& prepared,
        const CorrelatedBrownianStep<Dimension, TeamSize>& brownian_vector,
        philox::PhiloxKey key, std::uint64_t path,
        std::uint32_t step, unsigned int lane
    ) {
        const auto apply = [](
            std::size_t, const PreparedDynamics& model,
            const typename Dynamics::BrownianStep& shock,
            philox::DomainUniformSequence& auxiliary, State& state
        ) {
            Dynamics::one_step_transition(model, shock, auxiliary, state);
        };
        advance_one_step_with_transition<
            AssetOffset, FactorOffset, Dimension, TeamSize>(
                states, prepared, brownian_vector, key, path, step,
                lane, apply);
    }

    template<std::size_t AssetOffset, unsigned int TeamSize>
    __device__ __forceinline__ static float minimum_ratio(
        const States& states,
        const Prepared& prepared,
        unsigned int lane
    ) {
        float minimum = __int_as_float(0x7f800000);
        constexpr std::size_t lane_shift=AssetOffset%TeamSize;
        constexpr std::size_t local_count=(Count+TeamSize-1U)/TeamSize;
        for(std::size_t local=0U;local<local_count;++local) {
            const std::size_t i=local*TeamSize
                +(lane+TeamSize-lane_shift)%TeamSize;
            if(i<Count) {
                minimum=fminf(minimum,Dynamics::spot(states.model[i])
                    /prepared.initial_spot[i]);
            }
        }
        return minimum;
    }

    template<std::size_t AssetOffset, unsigned int TeamSize>
    __device__ __forceinline__ static float sum_ratio(
        const States& states, const Prepared& prepared, unsigned int lane
    ) {
        float sum = 0.0f;
        constexpr std::size_t lane_shift=AssetOffset%TeamSize;
        constexpr std::size_t local_count=(Count+TeamSize-1U)/TeamSize;
        for(std::size_t local=0U;local<local_count;++local) {
            const std::size_t i=local*TeamSize
                +(lane+TeamSize-lane_shift)%TeamSize;
            if(i<Count) {
                sum+=Dynamics::spot(states.model[i])
                    /prepared.initial_spot[i];
            }
        }
        return sum;
    }

    template<std::size_t AssetOffset, unsigned int TeamSize>
    __device__ __forceinline__ static float maximum_ratio(
        const States& states, const Prepared& prepared, unsigned int lane
    ) {
        float maximum = -__int_as_float(0x7f800000);
        constexpr std::size_t lane_shift = AssetOffset % TeamSize;
        constexpr std::size_t local_count = (Count + TeamSize - 1U) / TeamSize;
        for (std::size_t local = 0U; local < local_count; ++local) {
            const std::size_t i = local * TeamSize
                + (lane + TeamSize - lane_shift) % TeamSize;
            if (i < Count)
                maximum = fmaxf(maximum, Dynamics::spot(states.model[i])
                    / prepared.initial_spot[i]);
        }
        return maximum;
    }
};

template<typename... Groups>
struct MultiAssetPack;

template<>
struct MultiAssetPack<> {
    static constexpr std::size_t kAssetCount = 0U;
    static constexpr std::size_t kFactorCount = 0U;
    struct Inputs {};
    struct Prepared {};
    struct States {};
    template<std::size_t = 0U, std::size_t = 0U>
    __device__ __forceinline__ static void prepare_selected(
        Prepared&, const Inputs&, std::uint32_t, float, const float*
    ) {}
    template<std::size_t, unsigned int>
    __device__ __forceinline__ static void initial_states(
        States&, const Prepared&, unsigned int
    ) {}
    template<std::size_t, std::size_t, std::size_t Dimension,
             unsigned int TeamSize>
    __device__ __forceinline__ static void advance_one_step(
        States&, const Prepared&,
        const CorrelatedBrownianStep<Dimension, TeamSize>&,
        philox::PhiloxKey, std::uint64_t, std::uint32_t, unsigned int
    ) {}
    template<std::size_t, std::size_t, std::size_t Dimension,
             unsigned int TeamSize>
    __device__ __forceinline__ static void advance_one_step_with_rate(
        States&, const Prepared&,
        const CorrelatedBrownianStep<Dimension, TeamSize>&,
        philox::PhiloxKey, std::uint64_t, std::uint32_t,
        unsigned int, float
    ) {}
    template<std::size_t, unsigned int>
    __device__ __forceinline__ static float minimum_ratio(
        const States&, const Prepared&, unsigned int
    ) { return __int_as_float(0x7f800000); }
    template<std::size_t, unsigned int>
    __device__ __forceinline__ static float sum_ratio(
        const States&, const Prepared&, unsigned int
    ) { return 0.0f; }
    template<std::size_t, unsigned int>
    __device__ __forceinline__ static float maximum_ratio(
        const States&, const Prepared&, unsigned int
    ) { return -__int_as_float(0x7f800000); }
};

template<typename Head, typename... Tail>
struct MultiAssetPack<Head, Tail...> {
    using Rest = MultiAssetPack<Tail...>;
    static constexpr std::size_t kAssetCount =
        Head::kAssetCount + Rest::kAssetCount;
    static constexpr std::size_t kFactorCount =
        Head::kFactorCount + Rest::kFactorCount;
    static_assert(kAssetCount <= 254U);
    struct Inputs {
        typename Head::Inputs head;
        typename Rest::Inputs tail;
    };
    struct Prepared {
        typename Head::Prepared head;
        typename Rest::Prepared tail;
    };
    struct States {
        typename Head::States head;
        typename Rest::States tail;
    };

    template<std::size_t AssetOffset = 0U,
             std::size_t FactorOffset = 0U>
    __device__ __forceinline__ static void prepare_selected(
        Prepared& output, const Inputs& input, std::uint32_t row,
        float dt, const float* packed_lower
    ) {
        Head::template prepare_selected<FactorOffset>(
            output.head, input.head, row, dt, packed_lower);
        Rest::template prepare_selected<
            AssetOffset + Head::kAssetCount,
            FactorOffset + Head::kFactorCount
        >(output.tail, input.tail, row, dt, packed_lower);
    }

    template<std::size_t AssetOffset = 0U, unsigned int TeamSize = 1U>
    __device__ __forceinline__ static void initial_states(
        States& states, const Prepared& prepared, unsigned int lane
    ) {
        Head::template initial_states<AssetOffset, TeamSize>(
            states.head, prepared.head, lane
        );
        Rest::template initial_states<
            AssetOffset + Head::kAssetCount, TeamSize
        >(states.tail, prepared.tail, lane);
    }

    template<
        std::size_t AssetOffset = 0U, std::size_t FactorOffset = 0U,
        std::size_t Dimension = kFactorCount, unsigned int TeamSize = 1U
    >
    __device__ __forceinline__ static void advance_one_step(
        States& states,
        const Prepared& prepared,
        const CorrelatedBrownianStep<Dimension, TeamSize>& brownian_vector,
        philox::PhiloxKey key,
        std::uint64_t path,
        std::uint32_t step,
        unsigned int lane
    ) {
        Head::template advance_one_step<
            AssetOffset, FactorOffset, Dimension, TeamSize
        >(
            states.head, prepared.head, brownian_vector,
            key, path, step, lane
        );
        if constexpr (sizeof...(Tail) > 0U) {
            Rest::template advance_one_step<
                AssetOffset + Head::kAssetCount,
                FactorOffset + Head::kFactorCount,
                Dimension, TeamSize
            >(
                states.tail, prepared.tail, brownian_vector,
                key, path, step, lane
            );
        }
    }

    template<
        std::size_t AssetOffset = 0U, std::size_t FactorOffset = 0U,
        std::size_t Dimension = kFactorCount, unsigned int TeamSize = 1U
    >
    __device__ __forceinline__ static void advance_one_step_with_rate(
        States& states, const Prepared& prepared,
        const CorrelatedBrownianStep<Dimension, TeamSize>& brownian_vector,
        philox::PhiloxKey key, std::uint64_t path,
        std::uint32_t step, unsigned int lane,
        float rate_integral_step
    ) {
        Head::template advance_one_step_with_rate<
            AssetOffset, FactorOffset, Dimension, TeamSize>(
                states.head, prepared.head, brownian_vector, key,
                path, step, lane, rate_integral_step);
        if constexpr (sizeof...(Tail) > 0U) {
            Rest::template advance_one_step_with_rate<
                AssetOffset + Head::kAssetCount,
                FactorOffset + Head::kFactorCount,
                Dimension, TeamSize>(
                    states.tail, prepared.tail, brownian_vector, key,
                    path, step, lane, rate_integral_step);
        }
    }

    template<std::size_t AssetOffset = 0U, unsigned int TeamSize = 1U>
    __device__ __forceinline__ static float minimum_ratio(
        const States& states,
        const Prepared& prepared,
        unsigned int lane
    ) {
        const float head = Head::template minimum_ratio<
            AssetOffset, TeamSize
        >(states.head, prepared.head, lane);
        if constexpr (sizeof...(Tail) == 0U) {
            return head;
        } else {
            return fminf(
                head,
                Rest::template minimum_ratio<
                    AssetOffset + Head::kAssetCount, TeamSize
                >(states.tail, prepared.tail, lane)
            );
        }
    }

    template<std::size_t AssetOffset = 0U, unsigned int TeamSize = 1U>
    __device__ __forceinline__ static float sum_ratio(
        const States& states, const Prepared& prepared, unsigned int lane
    ) {
        const float head = Head::template sum_ratio<AssetOffset, TeamSize>(
            states.head, prepared.head, lane);
        if constexpr (sizeof...(Tail) == 0U) return head;
        else return head + Rest::template sum_ratio<
            AssetOffset + Head::kAssetCount, TeamSize
        >(states.tail, prepared.tail, lane);
    }

    template<std::size_t AssetOffset = 0U, unsigned int TeamSize = 1U>
    __device__ __forceinline__ static float maximum_ratio(
        const States& states, const Prepared& prepared, unsigned int lane
    ) {
        const float head = Head::template maximum_ratio<AssetOffset, TeamSize>(
            states.head, prepared.head, lane);
        if constexpr (sizeof...(Tail) == 0U) return head;
        else return fmaxf(head, Rest::template maximum_ratio<
            AssetOffset + Head::kAssetCount, TeamSize
        >(states.tail, prepared.tail, lane));
    }
};

}  // namespace ai_factory::workbench::simulation
