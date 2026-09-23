// Vanilla payoff evaluation on a frozen central exercise trace.
#pragma once

#include "common/equity/discount.cuh"
#include "common/option_side.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "common/price_gradients/result.cuh"
#include "product/american_option/frozen_exercise_value.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::product::american_option::price_gradients {

template<OptionSide Side, typename ReplayPolicy>
struct FrozenExerciseEvaluator {
    template<typename Scenario>
    struct Prepared {
        typename ReplayPolicy::Prepared replay;
        const Scenario* central;
        const Scenario* first;
        const Scenario* second;
        ::ai_factory::workbench::price_gradients::Stencil stencil;
        float central_initial_discount;
        float central_exercise_discount;
        float first_initial_discount;
        float first_exercise_discount;
        float second_initial_discount;
        float second_exercise_discount;
    };

    template<typename Scenario>
    __device__ __forceinline__ static Prepared<Scenario> prepare(
        const Scenario& central,
        const Scenario& first,
        const Scenario& second,
        const ::ai_factory::workbench::price_gradients::Stencil& stencil,
        float dt,
        float first_exercise_time,
        float exercise_interval,
        float central_initial_discount,
        float central_exercise_discount
    ) {
        return {
            ReplayPolicy::prepare(central, first, second, dt),
            &central,
            &first,
            &second,
            stencil,
            central_initial_discount,
            central_exercise_discount,
            first.model.risk_free_rate == central.model.risk_free_rate
                ? central_initial_discount
                : equity::constant_rate_discount_factor(
                    first.model, first_exercise_time
                ),
            first.model.risk_free_rate == central.model.risk_free_rate
                ? central_exercise_discount
                : equity::constant_rate_discount_factor(
                    first.model, exercise_interval
                ),
            second.model.risk_free_rate == central.model.risk_free_rate
                ? central_initial_discount
                : equity::constant_rate_discount_factor(
                    second.model, first_exercise_time
                ),
            second.model.risk_free_rate == central.model.risk_free_rate
                ? central_exercise_discount
                : equity::constant_rate_discount_factor(
                    second.model, exercise_interval
                ),
        };
    }

    __device__ __forceinline__ static float discounted_payoff(
        float spot,
        float strike,
        std::uint32_t observation,
        float initial_discount,
        float exercise_discount
    ) {
        float value = payoff::vanilla_option_payoff<Side>(spot, strike);
        // Match the central backward FP32 multiplication order exactly.
        for (std::uint32_t date = 0U; date < observation; ++date) {
            value = exercise_discount * value;
        }
        return initial_discount * value;
    }

    template<typename Scenario>
    __device__ __forceinline__ static float path_gradient(
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
        const bool central_payoff_shape = prepared.first->reuse_central
            && prepared.second->reuse_central
            && prepared.first->product.strike
                == prepared.central->product.strike
            && prepared.second->product.strike
                == prepared.central->product.strike
            && prepared.first->model.risk_free_rate
                == prepared.central->model.risk_free_rate
            && prepared.second->model.risk_free_rate
                == prepared.central->model.risk_free_rate;
        if (central_payoff_shape
            && prepared.stencil.kind
                == ::ai_factory::workbench::price_gradients::StencilKind::centered) {
            return centered_frozen_exercise_gradient<Side>(
                exercise,
                spots.first,
                spots.second,
                prepared.central->product.strike,
                prepared.central_initial_discount,
                prepared.central_exercise_discount,
                prepared.stencil.represented_width
            );
        }
        const float first_strike = central_payoff_shape
            ? prepared.central->product.strike
            : prepared.first->product.strike;
        const float second_strike = central_payoff_shape
            ? prepared.central->product.strike
            : prepared.second->product.strike;
        const float first_exercise_discount = central_payoff_shape
            ? prepared.central_exercise_discount
            : prepared.first_exercise_discount;
        const float second_exercise_discount = central_payoff_shape
            ? prepared.central_exercise_discount
            : prepared.second_exercise_discount;
        const float first_initial_discount = central_payoff_shape
            ? prepared.central_initial_discount
            : prepared.first_initial_discount;
        const float second_initial_discount = central_payoff_shape
            ? prepared.central_initial_discount
            : prepared.second_initial_discount;
        float first = payoff::vanilla_option_payoff<Side>(
            spots.first, first_strike
        );
        float second = payoff::vanilla_option_payoff<Side>(
            spots.second, second_strike
        );
        // Advance both paired values together. Besides avoiding duplicate
        // loop control, this preserves the historical price-delta FP32 order
        // when both endpoints share the central discount curve.
        for (std::uint32_t date = 0U;
             date < exercise.observation;
             ++date) {
            first = first_exercise_discount * first;
            second = second_exercise_discount * second;
        }
        first = first_initial_discount * first;
        second = second_initial_discount * second;
        if (prepared.stencil.kind
            == ::ai_factory::workbench::price_gradients::StencilKind::centered) {
            return (second - first) / prepared.stencil.represented_width;
        }
        const float central = discounted_payoff(
            exercise.spot,
            prepared.central->product.strike,
            exercise.observation,
            prepared.central_initial_discount,
            prepared.central_exercise_discount
        );
        return ::ai_factory::workbench::price_gradients::difference(
            prepared.stencil, central, first, second
        );
    }

    template<typename Scenario>
    __device__ __forceinline__ static float initial_gradient(
        const Prepared<Scenario>& prepared
    ) {
        const float central = payoff::vanilla_option_payoff<Side>(
            prepared.central->model.spot,
            prepared.central->product.strike
        );
        const float first = payoff::vanilla_option_payoff<Side>(
            prepared.first->model.spot,
            prepared.first->product.strike
        );
        const float second = payoff::vanilla_option_payoff<Side>(
            prepared.second->model.spot,
            prepared.second->product.strike
        );
        return ::ai_factory::workbench::price_gradients::difference(
            prepared.stencil, central, first, second
        );
    }
};

}  // namespace ai_factory::workbench::product::american_option::price_gradients
