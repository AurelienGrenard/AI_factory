// American payoff reconstruction on row-local sensitivity nodes.
#pragma once

#include "common/equity/discount.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_frozen_exercise_nodes.cuh"
#include "common/option_side.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "common/price_gradients/reconstruction.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product::american_option::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<
    OptionSide Side,
    ::ai_factory::workbench::price_gradients::SensitivityOrders Orders,
    typename ReplayPolicy
>
struct DevicePreparedFrozenExerciseEvaluator {
    static constexpr std::size_t kNodeCapacity =
        pg::SensitivityTraits<Orders>::node_capacity;

    template<typename Scenario>
    struct Prepared {
        typename ReplayPolicy::Prepared replay;
        pg::SensitivityStencil<kNodeCapacity> stencil;
        float strikes[kNodeCapacity]{};
        float initial_spots[kNodeCapacity]{};
        float initial_discounts[kNodeCapacity]{};
        float exercise_discounts[kNodeCapacity]{};
        std::uint8_t node_count = 0U;
    };

    template<typename Scenario>
    __device__ __forceinline__ static Prepared<Scenario> prepare(
        const pg::SensitivityTask<Scenario, kNodeCapacity>& task,
        float dt,
        float first_exercise_time,
        float exercise_interval,
        float central_initial_discount,
        float central_exercise_discount
    ) {
        Prepared<Scenario> result{};
        result.stencil = task.stencil;
        result.node_count = static_cast<std::uint8_t>(
            pg::active_node_count(task.stencil)
        );
        result.replay = ReplayPolicy::prepare(
            task.nodes,
            result.node_count,
            longstaff_schwartz::price_gradients::FrozenExerciseReplayTime{
                dt,
                first_exercise_time,
                exercise_interval,
            }
        );
        const float central_rate = task.nodes[0U].model.risk_free_rate;
        #pragma unroll
        for (std::size_t node = 0U; node < kNodeCapacity; ++node) {
            if (node >= result.node_count) continue;
            result.strikes[node] = task.nodes[node].product.strike;
            result.initial_spots[node] = task.nodes[node].model.spot;
            if (task.nodes[node].model.risk_free_rate == central_rate) {
                result.initial_discounts[node] = central_initial_discount;
                result.exercise_discounts[node] = central_exercise_discount;
            } else {
                result.initial_discounts[node] =
                    equity::constant_rate_discount_factor(
                        task.nodes[node].model, first_exercise_time
                    );
                result.exercise_discounts[node] =
                    equity::constant_rate_discount_factor(
                        task.nodes[node].model, exercise_interval
                    );
            }
        }
        return result;
    }

    template<typename Scenario>
    __device__ __forceinline__ static pg::SensitivityResult evaluate_path(
        const Prepared<Scenario>& prepared,
        longstaff_schwartz::FrozenExerciseTrace exercise,
        philox::PhiloxKey key,
        std::size_t path,
        std::uint32_t initial_transition_count,
        std::uint32_t transitions_per_exercise
    ) {
        const auto spots = ReplayPolicy::evaluate(
            prepared.replay,
            exercise,
            key,
            path,
            initial_transition_count,
            transitions_per_exercise
        );
        pg::SensitivityValues<kNodeCapacity> values{};
        #pragma unroll
        for (std::size_t node = 0U; node < kNodeCapacity; ++node) {
            if (node < prepared.node_count) {
                values[node] = payoff::vanilla_option_payoff<Side>(
                    spots[node], prepared.strikes[node]
                );
            }
        }
        // Advance every represented node in lockstep to preserve the central
        // LSM multiplication order and common exercise observation.
        for (std::uint32_t date = 0U;
             date < exercise.observation;
             ++date) {
            #pragma unroll
            for (std::size_t node = 0U; node < kNodeCapacity; ++node) {
                if (node < prepared.node_count) {
                    values[node] = prepared.exercise_discounts[node]
                        * values[node];
                }
            }
        }
        #pragma unroll
        for (std::size_t node = 0U; node < kNodeCapacity; ++node) {
            if (node < prepared.node_count) {
                values[node] = prepared.initial_discounts[node]
                    * values[node];
            }
        }
        return pg::reconstruct_sensitivity<Orders>(
            prepared.stencil, values
        );
    }

    template<typename Scenario>
    __device__ __forceinline__ static pg::SensitivityResult evaluate_initial(
        const Prepared<Scenario>& prepared
    ) {
        pg::SensitivityValues<kNodeCapacity> values{};
        #pragma unroll
        for (std::size_t node = 0U; node < kNodeCapacity; ++node) {
            if (node < prepared.node_count) {
                values[node] = payoff::vanilla_option_payoff<Side>(
                    prepared.initial_spots[node],
                    prepared.strikes[node]
                );
            }
        }
        return pg::reconstruct_sensitivity<Orders>(
            prepared.stencil, values
        );
    }
};

}  // namespace ai_factory::workbench::product::american_option::price_gradients
