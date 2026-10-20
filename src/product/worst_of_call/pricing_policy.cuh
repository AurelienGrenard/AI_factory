#pragma once

#include "common/equity/multi_asset_path_product_policy.cuh"
#include "product/worst_of_call/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::product {

struct WorstOfCallPathPolicy {
    static constexpr bool kTerminalCashflow = true;
    using ProductParameters = WorstOfCallParameters;

    struct PreparedProduct {
        float strike;
        float discount_factor;
    };

    struct Handler {
        float terminal_minimum_ratio = 0.0f;
        __device__ __forceinline__ bool on_observation(
            std::uint32_t, float value
        ) {
            terminal_minimum_ratio = value;
            return true;
        }
    };

    __device__ __forceinline__ static PreparedProduct prepare(
        const ProductParameters& product, float discount_factor
    ) {
        return {product.strike, discount_factor};
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct&
    ) {
        return {};
    }

    __device__ __forceinline__ static bool is_observation_step(
        std::uint32_t step, std::uint32_t total_steps,
        std::uint32_t, const ProductParameters&
    ) {
        return step + 1U == total_steps;
    }

    template<typename Pack, unsigned int TeamSize>
    __device__ __forceinline__ static float observe(
        const typename Pack::States& states,
        const typename Pack::Prepared& prepared,
        unsigned int lane
    ) {
        return equity::team_minimum_spot_ratio<Pack, TeamSize>(
            states, prepared, lane
        );
    }

    template<typename MarkovPack, typename RoughPolicy>
    __device__ __forceinline__ static float observe_joint(
        const typename MarkovPack::States& markov_states,
        const typename MarkovPack::Prepared& markov_prepared,
        float rough_log_spot,
        float rough_initial_log_spot
    ) {
        const float markov_minimum =
            MarkovPack::template minimum_ratio<0U, 1U>(
                markov_states, markov_prepared, 0U
            );
        return fminf(markov_minimum,
            expf(rough_log_spot - rough_initial_log_spot));
    }

    template<typename MarkovPack, typename RoughPolicy,
             std::size_t RoughCount, unsigned TeamSize=1U>
    __device__ __forceinline__ static float observe_joint_group(
        const typename MarkovPack::States& markov_states,
        const typename MarkovPack::Prepared& markov_prepared,
        const float (&rough_log_spots)[RoughCount],
        const float (&rough_initial_logs)[RoughCount],
        unsigned lane=0U
    ) {
        return equity::team_joint_minimum_spot_ratio<
            MarkovPack,RoughCount,TeamSize>(
                markov_states,markov_prepared,
                rough_log_spots,rough_initial_logs,lane);
    }

    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product, const Handler& handler
    ) {
        return product.discount_factor
            * fmaxf(handler.terminal_minimum_ratio - product.strike, 0.0f);
    }
};

}  // namespace ai_factory::workbench::product
