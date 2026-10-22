// Shared terminal call contract for an initial-spot-normalized basket.
#pragma once

#include "common/equity/multi_asset_path_product_policy.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product::detail {

template<typename Parameters, equity::BasketSpotAggregation Aggregation>
struct TerminalBasketCallPathPolicy {
    static constexpr bool kTerminalCashflow = true;
    using ProductParameters = Parameters;

    struct PreparedProduct {
        float strike;
        float discount_factor;
    };

    struct Handler {
        float terminal_basket_ratio = 0.0f;

        __device__ __forceinline__ bool on_observation(
            std::uint32_t, float basket_ratio
        ) {
            terminal_basket_ratio = basket_ratio;
            return true;
        }
    };

    __host__ static bool valid_parameters(
        const ProductParameters& product, std::uint32_t
    ) {
        return std::isfinite(product.strike)
            && product.strike >= 0.0f
            && product.maturity_days > 0U;
    }

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

    __host__ __device__ static bool is_observation_step(
        std::uint32_t step, std::uint32_t total_steps,
        std::uint32_t, const ProductParameters&
    ) {
        return step + 1U == total_steps;
    }

    template<typename Pack, unsigned TeamSize>
    __device__ __forceinline__ static float observe(
        const typename Pack::States& states,
        const typename Pack::Prepared& prepared,
        unsigned lane
    ) {
        return equity::team_basket_spot_ratio<
            Aggregation, Pack, TeamSize>(states, prepared, lane);
    }

    template<typename MarkovPack, typename RoughPack,
             std::size_t RoughCount, unsigned TeamSize = 1U>
    __device__ __forceinline__ static float observe_joint_group(
        const typename MarkovPack::States& markov_states,
        const typename MarkovPack::Prepared& markov_prepared,
        const float (&rough_log_spots)[RoughCount],
        const float (&rough_initial_logs)[RoughCount],
        unsigned lane = 0U
    ) {
        return equity::team_joint_basket_spot_ratio<
            Aggregation, MarkovPack, RoughCount, TeamSize>(
                markov_states, markov_prepared,
                rough_log_spots, rough_initial_logs, lane);
    }

    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product, const Handler& handler
    ) {
        return product.discount_factor * fmaxf(
            handler.terminal_basket_ratio - product.strike, 0.0f);
    }
};

}  // namespace ai_factory::workbench::product::detail
