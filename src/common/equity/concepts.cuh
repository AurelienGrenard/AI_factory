// Compile-time contracts for composable equity Monte Carlo policies.
#pragma once

#include "common/philox.cuh"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::equity {

inline constexpr std::size_t kMaximumPreparedRowBytes = 256U;
inline constexpr std::size_t kMaximumObservationHandlerBytes = 128U;

// Every standard equity dynamics exposes one prepared homogeneous transition,
// one path-local random context, one mutable state and a spot accessor.
template<typename Dynamics>
concept EquityDynamicsPolicy =
    std::is_trivially_copyable_v<typename Dynamics::Parameters>
    && std::is_trivially_copyable_v<typename Dynamics::PreparedDynamics>
    && std::is_trivially_copyable_v<typename Dynamics::RandomContext>
    && std::is_trivially_copyable_v<typename Dynamics::State>
    && std::constructible_from<
        typename Dynamics::RandomContext,
        philox::PhiloxKey,
        std::uint64_t
    >
    && requires(
        const typename Dynamics::Parameters& parameters,
        const typename Dynamics::PreparedDynamics& dynamics,
        typename Dynamics::RandomContext& random,
        typename Dynamics::State& state,
        const typename Dynamics::State& const_state,
        std::uint32_t step_count,
        float delta_t
    ) {
        typename Dynamics::Parameters;
        typename Dynamics::PreparedDynamics;
        typename Dynamics::RandomContext;
        typename Dynamics::State;

        {
            Dynamics::prepare_dynamics(parameters, delta_t)
        } -> std::same_as<typename Dynamics::PreparedDynamics>;
        {
            Dynamics::initial_state(dynamics)
        } -> std::same_as<typename Dynamics::State>;
        {
            Dynamics::simulate_one_step(dynamics, random, state)
        } -> std::same_as<void>;
        {
            Dynamics::advance(dynamics, step_count, random, state)
        } -> std::same_as<void>;
        { Dynamics::spot(const_state) } -> std::same_as<float>;
    };

// Optional capability used only by products accumulating logarithmic spots.
template<typename Dynamics>
concept SupportsLogSpot =
    EquityDynamicsPolicy<Dynamics>
    && requires(const typename Dynamics::State& state) {
        { Dynamics::log_spot(state) } -> std::same_as<float>;
    };

// Optional capability used by the current constant-rate discount policy.
template<typename Dynamics>
concept SupportsRiskFreeRate =
    EquityDynamicsPolicy<Dynamics>
    && requires(const typename Dynamics::Parameters& parameters) {
        { Dynamics::risk_free_rate(parameters) } -> std::same_as<float>;
    };

// Direct-transition models keep invariant coefficients separate from the
// coefficients prepared for one interval.
template<typename Dynamics>
concept ExactTransitionDynamicsPolicy =
    EquityDynamicsPolicy<Dynamics>
    && std::is_trivially_copyable_v<typename Dynamics::PreparedModel>
    && std::is_trivially_copyable_v<typename Dynamics::PreparedTransition>
    && requires(
        const typename Dynamics::Parameters& parameters,
        const typename Dynamics::PreparedModel& model,
        const typename Dynamics::PreparedTransition& transition,
        typename Dynamics::RandomContext& random,
        typename Dynamics::State& state,
        float delta_t
    ) {
        typename Dynamics::PreparedModel;
        typename Dynamics::PreparedTransition;

        {
            Dynamics::prepare_model(parameters)
        } -> std::same_as<typename Dynamics::PreparedModel>;
        {
            Dynamics::prepare_transition(model, delta_t)
        } -> std::same_as<typename Dynamics::PreparedTransition>;
        {
            Dynamics::initial_state(model)
        } -> std::same_as<typename Dynamics::State>;
        {
            Dynamics::simulate_one_step(
                model,
                transition,
                random,
                state
            )
        } -> std::same_as<void>;
    };

// A handler receives each contractual observation and decides whether the
// path must continue. It is statically dispatched and remains path-local.
template<typename Handler, typename Dynamics>
concept ObservationHandlerFor =
    EquityDynamicsPolicy<Dynamics>
    && std::is_trivially_copyable_v<Handler>
    && sizeof(Handler) <= kMaximumObservationHandlerBytes
    && requires(
        Handler& handler,
        std::uint32_t observation,
        const typename Dynamics::State& state
    ) {
        {
            handler.on_initial_state(state)
        } -> std::same_as<bool>;
        {
            handler.on_observation(observation, state)
        } -> std::same_as<bool>;
    };

template<EquityDynamicsPolicy Dynamics>
struct ObservationHandlerProbe {
    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t,
        const typename Dynamics::State&
    ) {
        return true;
    }
};

// A schedule owns time conversion and path advancement, but no payoff logic.
template<typename Schedule>
concept EquitySchedulePolicy =
    EquityDynamicsPolicy<typename Schedule::Dynamics>
    && std::is_trivially_copyable_v<typename Schedule::Configuration>
    && std::is_trivially_copyable_v<typename Schedule::DeviceInputs>
    && std::is_trivially_copyable_v<typename Schedule::Definition>
    && std::is_trivially_copyable_v<typename Schedule::PreparedSchedule>
    && requires(
        const typename Schedule::Dynamics::Parameters& parameters,
        const typename Schedule::Definition& definition,
        const typename Schedule::Configuration& configuration,
        const typename Schedule::DeviceInputs& inputs,
        const typename Schedule::PreparedSchedule& prepared,
        philox::PhiloxKey key,
        std::size_t path,
        std::size_t monte_carlo_paths_per_price,
        ObservationHandlerProbe<typename Schedule::Dynamics>& handler
    ) {
        typename Schedule::Dynamics;
        typename Schedule::Configuration;
        typename Schedule::DeviceInputs;
        typename Schedule::Definition;
        typename Schedule::PreparedSchedule;

        {
            Schedule::prepare(
                parameters,
                definition,
                configuration,
                inputs
            )
        } -> std::same_as<typename Schedule::PreparedSchedule>;
        {
            Schedule::interval_year_fraction(
                definition,
                0U,
                configuration
            )
        } -> std::same_as<float>;
        {
            Schedule::total_year_fraction(definition, configuration)
        } -> std::same_as<float>;
        {
            Schedule::observation_count(prepared)
        } -> std::same_as<std::uint32_t>;
        {
            Schedule::simulate(prepared, key, path, handler)
        } -> std::same_as<typename Schedule::Dynamics::State>;
        {
            Schedule::validate_configuration(
                configuration,
                inputs,
                monte_carlo_paths_per_price
            )
        } -> std::same_as<void>;
    };

// One pricing policy binds a product to a schedule and exposes the minimal
// interface consumed by the generic one-block-per-price kernel.
template<typename Pricing>
concept ScalarMonteCarloPricingPolicy =
    EquityDynamicsPolicy<typename Pricing::Dynamics>
    && std::same_as<
        typename Pricing::ModelParameters,
        typename Pricing::Dynamics::Parameters
    >
    && std::is_trivially_copyable_v<typename Pricing::ProductParameters>
    && std::is_trivially_copyable_v<typename Pricing::PricingConfiguration>
    && std::is_trivially_copyable_v<typename Pricing::DeviceInputs>
    && std::is_trivially_copyable_v<typename Pricing::PreparedRow>
    && sizeof(typename Pricing::PreparedRow) <= kMaximumPreparedRowBytes
    && requires(
        const typename Pricing::ModelParameters& model,
        const typename Pricing::ProductParameters& product,
        const typename Pricing::PricingConfiguration& configuration,
        const typename Pricing::DeviceInputs& inputs,
        const typename Pricing::PreparedRow& row,
        std::uint64_t seed,
        std::size_t path,
        std::size_t monte_carlo_paths_per_price
    ) {
        typename Pricing::Dynamics;
        typename Pricing::ModelParameters;
        typename Pricing::ProductParameters;
        typename Pricing::PricingConfiguration;
        typename Pricing::DeviceInputs;
        typename Pricing::PreparedRow;

        {
            Pricing::prepare_row(
                model,
                product,
                configuration,
                inputs,
                seed
            )
        } -> std::same_as<typename Pricing::PreparedRow>;
        {
            Pricing::evaluate_path(row, path)
        } -> std::same_as<float>;
        {
            Pricing::validate_configuration(
                configuration,
                inputs,
                monte_carlo_paths_per_price
            )
        } -> std::same_as<void>;
    };

}  // namespace ai_factory::workbench::equity
