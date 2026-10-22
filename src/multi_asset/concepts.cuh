// Contracts for correlated multi-asset transitions.
#pragma once

#include "dynamics/concepts.cuh"

#include <concepts>
#include <cstdint>

namespace ai_factory::workbench::simulation {

// A correlated-step model receives physical Brownian drivers from the joint
// engine. Auxiliary laws use a separate path/step/model Philox domain.
template<typename Dynamics, typename AuxiliaryRandom>
concept CorrelatedStepDynamicsPolicy =
    FixedStepDynamicsPolicy<Dynamics>
    && requires {
        typename Dynamics::BrownianStep;
        { Dynamics::kBrownianDimension } -> std::convertible_to<unsigned int>;
    }
    && requires(
        const typename Dynamics::PreparedDynamics& prepared,
        typename Dynamics::BrownianStep brownian,
        AuxiliaryRandom& auxiliary,
        typename Dynamics::State& state
    ) {
        {
            Dynamics::one_step_transition(
                prepared, brownian, auxiliary, state
            )
        } -> std::same_as<void>;
    };

// A correlated equity factor declares how a short-rate integral changes its
// own transition. The scheduler never inspects model state or drift fields.
template<typename Dynamics, typename AuxiliaryRandom>
concept RateAdjustedCorrelatedStepDynamicsPolicy =
    CorrelatedStepDynamicsPolicy<Dynamics, AuxiliaryRandom>
    && requires(
        const typename Dynamics::PreparedDynamics& prepared,
        typename Dynamics::BrownianStep brownian,
        AuxiliaryRandom& auxiliary,
        typename Dynamics::State& state,
        float carry_shift
    ) {
        { Dynamics::one_step_transition_with_rate(
            prepared, brownian, auxiliary, state, carry_shift)
        } -> std::same_as<void>;
    };

// A rough spot declares how a shared short-rate integral enters its scan.
template<typename Policy>
concept RateAdjustedRoughSpotPolicy = requires(
    const typename Policy::PreparedModel& prepared,
    float volatility, float normal,
    float& log_spot, float carry_shift
) {
    { Policy::advance_spot_cell_with_rate(
        prepared, volatility, normal, log_spot, carry_shift)
    } -> std::same_as<void>;
};

}  // namespace ai_factory::workbench::simulation
