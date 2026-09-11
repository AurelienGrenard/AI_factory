// Last-exercise numeraire adapter for the shared Bermudan payoff and LSM engine.
// Cashflows are stored as H(t) * P(0,T*) / P(t,T*): no interval discount is needed.
#pragma once

#include "product/bermudan_swaption/pricing_policy.cuh"

namespace ai_factory::workbench::product {

template<typename BasePricingPolicy, typename BondAnalytics>
struct TerminalForwardBermudanSwaptionPricingPolicy : BasePricingPolicy {
    using Base = BasePricingPolicy;
    using typename Base::PreparedRow;
    using typename Base::Schedule;
    using typename Base::Dynamics;
    using typename Base::ContinuationState;
    using Observation = typename Schedule::Observation;
    struct StateView : Base::StateView { Observation* observations; };

    static std::vector<longstaff_schwartz::StateFieldDescriptor> observation_field_descriptors() {
        return {{sizeof(Observation), alignof(Observation)}};
    }
    static StateView make_state_view(
        unsigned char* workspace, const longstaff_schwartz::WorkspaceLayout& layout
    ) {
        if (layout.observation_fields.size() != 1U)
            throw std::logic_error("A terminal-forward policy requires an observation table.");
        return {Base::make_state_view(workspace, layout),
            longstaff_schwartz::workspace_pointer<Observation>(workspace, layout.observation_fields[0])};
    }

    __device__ __forceinline__ static void prepare_observations(PreparedRow& row, StateView states) {
        Observation* observations = states.observations + row.state_offset / row.paths_per_price;
        row.schedule.observations = observations;
        const auto count = row.product.exercise_count;
        const auto interval_days = row.product.payment_interval_days;
        const auto terminal_days = row.product.first_exercise_time_days + (count - 1U) * interval_days;
        row.schedule.initial_log_numeraire = BondAnalytics::log_zero_coupon_bond(
            row.analytics, Dynamics::initial_state(row.schedule.model), 0.0f,
            static_cast<float>(terminal_days) * row.time_day_fraction
        );
        for (std::uint32_t exercise = 0U; exercise < count; ++exercise) {
            const float interval = static_cast<float>(exercise == 0U
                ? row.product.first_exercise_time_days : interval_days) * row.time_day_fraction;
            const float remaining = static_cast<float>((count - 1U - exercise) * interval_days)
                * row.time_day_fraction;
            const float exercise_time = static_cast<float>(
                row.product.first_exercise_time_days + exercise * interval_days
            ) * row.time_day_fraction;
            const float terminal_time = static_cast<float>(terminal_days) * row.time_day_fraction;
            observations[exercise] = {
                Dynamics::prepare_transition(row.schedule.parameters, interval, remaining),
                BondAnalytics::log_A(row.analytics, exercise_time, terminal_time),
                BondAnalytics::B(row.analytics, exercise_time, terminal_time),
            };
        }
    }

    __device__ __forceinline__ static float normalized_payoff(
        const PreparedRow& row, float state, std::uint32_t exercise
    ) {
        const auto& observation = row.schedule.observations[exercise];
        const float log_ratio = fmaf(observation.numeraire_b, state,
            row.schedule.initial_log_numeraire - observation.log_numeraire_a);
        return Base::immediate_value_at(row, state, exercise) * expf(log_ratio);
    }
    __device__ __forceinline__ static float simulate_path(
        const PreparedRow& row, std::size_t path, std::size_t paths, StateView states
    ) {
        auto writer = ContinuationState::make_writer(
            states, row.state_offset + path, paths, row.product.exercise_count
        );
        return normalized_payoff(row, Schedule::simulate(row.schedule, row.key, path, writer),
            row.product.exercise_count - 1U);
    }
    __device__ __forceinline__ static float immediate_value(
        const PreparedRow& row, StateView states, std::size_t index
    ) {
        return normalized_payoff(row, ContinuationState::factor(states, index),
            Base::exercise_from_state_index(row, index));
    }
    __device__ __forceinline__ static double regression_target(
        const PreparedRow&, StateView, std::size_t, float value
    ) { return static_cast<double>(value); }
    __device__ __forceinline__ static float continued_cashflow(
        const PreparedRow&, StateView, std::size_t, float value
    ) { return value; }
    __device__ __forceinline__ static float initial_continuation_value(
        const PreparedRow&, StateView, std::size_t, std::size_t, float value
    ) { return value; }
};

}  // namespace ai_factory::workbench::product
