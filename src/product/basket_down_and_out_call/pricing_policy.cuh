#pragma once

#include "multi_asset/equity_basket_observation.cuh"
#include "product/basket_down_and_out_call/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product {

struct BasketDownAndOutCallPathPolicy {
    static constexpr bool kTerminalCashflow = true;
    using ProductParameters = BasketDownAndOutCallParameters;

    struct PreparedProduct {
        float strike;
        float lower_barrier;
        float discount_factor;
    };
    struct Handler {
        float lower_barrier;
        float terminal_basket = 0.0f;
        bool knocked_out = false;

        __device__ __forceinline__ bool on_observation(
            std::uint32_t, float basket
        ) {
            terminal_basket = basket;
            knocked_out |= basket <= lower_barrier;
            return !knocked_out;
        }
    };

    __host__ static bool valid_parameters(
        const ProductParameters& product, std::uint32_t steps_per_day
    ) {
        return std::isfinite(product.strike)
            && std::isfinite(product.lower_barrier)
            && product.lower_barrier > 0.0f
            && product.maturity_days > 0U
            && product.monitoring_interval_days > 0U
            && product.monitoring_interval_days <= 8192U
            && steps_per_day > 0U
            && static_cast<std::uint64_t>(steps_per_day)
                * product.monitoring_interval_days <= 0xffffffffULL;
    }

    __host__ __device__ static std::uint32_t observation_stride(
        std::uint32_t steps_per_day, const ProductParameters& product
    ) {
        return steps_per_day * product.monitoring_interval_days;
    }

    __host__ __device__ static bool is_observation_step(
        std::uint32_t step, std::uint32_t total_steps,
        std::uint32_t steps_per_day, const ProductParameters& product
    ) {
        const auto completed = step + 1U;
        return completed == total_steps
            || completed % observation_stride(steps_per_day, product) == 0U;
    }

    __device__ __forceinline__ static PreparedProduct prepare(
        const ProductParameters& product, float discount_factor
    ) {
        return {product.strike, product.lower_barrier, discount_factor};
    }
    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct& product
    ) { return {product.lower_barrier}; }

    template<typename Pack, unsigned int TeamSize>
    __device__ __forceinline__ static float observe(
        const typename Pack::States& states,
        const typename Pack::Prepared& prepared, unsigned int lane
    ) {
        return equity::team_basket_spot_ratio<
            equity::BasketSpotAggregation::ArithmeticMean,
            Pack, TeamSize>(states, prepared, lane);
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
        return equity::team_joint_basket_spot_ratio<
            equity::BasketSpotAggregation::ArithmeticMean,
            MarkovPack, RoughCount, TeamSize>(
                markov_states, markov_prepared,
                rough_log_spots, rough_initial_logs, lane);
    }

    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product, const Handler& handler
    ) {
        return handler.knocked_out ? 0.0f
            : product.discount_factor * fmaxf(
                handler.terminal_basket - product.strike, 0.0f
            );
    }
};

}  // namespace ai_factory::workbench::product
