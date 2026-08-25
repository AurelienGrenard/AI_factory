// Reusable CUDA runtime checks for statically composed dynamics policies.
#pragma once

#include "common/simulation/path_simulation.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::test {

template<typename Value>
__device__ __forceinline__ bool bitwise_equal(
    const Value& left,
    const Value& right
) {
    const auto* left_bytes =
        reinterpret_cast<const unsigned char*>(&left);
    const auto* right_bytes =
        reinterpret_cast<const unsigned char*>(&right);
    for (std::size_t byte = 0U; byte < sizeof(Value); ++byte) {
        if (left_bytes[byte] != right_bytes[byte]) return false;
    }
    return true;
}

// Check deterministic replay, advance(n) equivalence and path isolation.
template<
    simulation::FixedStepDynamicsPolicy Dynamics,
    typename StateInspector
>
__device__ __forceinline__ std::uint32_t
test_fixed_step_dynamics_contract(
    const typename Dynamics::Parameters& parameters,
    float delta_t,
    std::uint32_t transition_count,
    philox::PhiloxKey key,
    std::size_t path
) {
    const typename Dynamics::PreparedDynamics dynamics =
        Dynamics::prepare_dynamics(parameters, delta_t);
    const typename Dynamics::State initial =
        Dynamics::initial_state(dynamics);

    typename Dynamics::RandomContext batched_random(
        key, static_cast<std::uint64_t>(path)
    );
    typename Dynamics::State batched = initial;
    Dynamics::advance(
        dynamics, transition_count, batched_random, batched
    );

    typename Dynamics::RandomContext repeated_random(
        key, static_cast<std::uint64_t>(path)
    );
    typename Dynamics::State repeated = initial;
    for (std::uint32_t step = 0U; step < transition_count; ++step) {
        Dynamics::advance(dynamics, 1U, repeated_random, repeated);
    }

    const typename Dynamics::State first =
        simulation::simulate_fixed_step_terminal<Dynamics>(
            dynamics, transition_count, key, path
        );
    static_cast<void>(
        simulation::simulate_fixed_step_terminal<Dynamics>(
            dynamics, transition_count, key, path + 1U
        )
    );
    const typename Dynamics::State replay =
        simulation::simulate_fixed_step_terminal<Dynamics>(
            dynamics, transition_count, key, path
        );

    std::uint32_t result = 0U;
    result |= StateInspector::finite(initial) ? 1U : 0U;
    result |= bitwise_equal(batched, repeated) ? 2U : 0U;
    result |= bitwise_equal(first, replay) ? 4U : 0U;
    result |= StateInspector::finite(first) ? 8U : 0U;
    return result;
}

// Check exact-transition replay, one-observation parity and path isolation.
template<
    simulation::ExactTransitionDynamicsPolicy Dynamics,
    typename StateInspector
>
__device__ __forceinline__ std::uint32_t
test_exact_transition_dynamics_contract(
    const typename Dynamics::Parameters& parameters,
    float delta_t,
    philox::PhiloxKey key,
    std::size_t path
) {
    const typename Dynamics::PreparedModel model =
        Dynamics::prepare_model(parameters);
    const typename Dynamics::PreparedTransition transition =
        Dynamics::prepare_transition(model, delta_t);
    const typename Dynamics::State first =
        simulation::simulate_exact_transition_terminal<Dynamics>(
            model, transition, key, path
        );
    const typename Dynamics::State replay =
        simulation::simulate_exact_transition_terminal<Dynamics>(
            model, transition, key, path
        );

    simulation::ObservationHandlerProbe<Dynamics> handler;
    const typename Dynamics::State observed =
        simulation::simulate_exact_transition_regular_schedule<Dynamics>(
            model, transition, 1U, key, path, handler
        );
    static_cast<void>(
        simulation::simulate_exact_transition_terminal<Dynamics>(
            model, transition, key, path + 1U
        )
    );
    const typename Dynamics::State after_other_path =
        simulation::simulate_exact_transition_terminal<Dynamics>(
            model, transition, key, path
        );

    std::uint32_t result = 0U;
    result |= bitwise_equal(first, replay) ? 1U : 0U;
    result |= bitwise_equal(first, observed) ? 2U : 0U;
    result |= bitwise_equal(first, after_other_path) ? 4U : 0U;
    result |= StateInspector::finite(first) ? 8U : 0U;
    return result;
}

// Every exact rate model also exposes the same interval as one fixed step.
template<
    simulation::FixedStepDynamicsPolicy Dynamics,
    typename StateInspector
>
requires simulation::ExactTransitionDynamicsPolicy<Dynamics>
__device__ __forceinline__ bool test_exact_fixed_step_parity(
    const typename Dynamics::Parameters& parameters,
    float delta_t,
    philox::PhiloxKey key,
    std::size_t path
) {
    const typename Dynamics::PreparedDynamics dynamics =
        Dynamics::prepare_dynamics(parameters, delta_t);
    const typename Dynamics::State fixed =
        simulation::simulate_fixed_step_terminal<Dynamics>(
            dynamics, 1U, key, path
        );
    const typename Dynamics::PreparedModel model =
        Dynamics::prepare_model(parameters);
    const typename Dynamics::PreparedTransition transition =
        Dynamics::prepare_transition(model, delta_t);
    const typename Dynamics::State exact =
        simulation::simulate_exact_transition_terminal<Dynamics>(
            model, transition, key, path
        );
    return bitwise_equal(fixed, exact)
        && StateInspector::finite(fixed);
}

}  // namespace ai_factory::workbench::test
