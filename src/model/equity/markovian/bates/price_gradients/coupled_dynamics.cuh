// Public Bates QE-M and event-coupling adapter for terminal sensitivities.
#pragma once

#include "model/equity/markovian/bates/dynamics.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::bates::price_gradients {

struct CoupledDynamics {
    using ModelParameters = bates::ModelParameters;
    using State = bates::State;
    using Prepared = bates::PreparedModel;
    using RandomContext = DynamicsPolicy::RandomContext;

    struct Innovations {
        float variance_normal;
        float variance_uniform;
        float stock_normal;
        std::uint32_t jump_count;
        float jump_standard_normal_sum;
    };

    struct ContinuousInnovations {
        float variance_normal;
        float variance_uniform;
        float stock_normal;
    };

    struct TerminalAdjustment {
        std::uint32_t jump_count;
        float jump_standard_normal_sum;
    };

    static constexpr bool kExactTerminal = false;
    static constexpr bool kDrawRequiresCentralPrepared = true;
    // The distributed graph reuses the canonical continuous replay and
    // applies one coupled jump adjustment after the complete interval.
    static constexpr bool kHasDistributedTerminalAggregation = true;

    __device__ static Prepared prepare(const ModelParameters&, float);
    __device__ static State initial(const Prepared&);
    __device__ static Innovations draw(RandomContext&, const Prepared&);
    __device__ static ContinuousInnovations draw_continuous(
        RandomContext&, const Prepared&
    );
    __device__ static void transition_continuous(
        const Prepared&, const ContinuousInnovations&, State&
    );
    template<std::size_t NodeCapacity, typename StepCount>
    __device__ static void draw_terminal_adjustments(
        RandomContext&,
        std::uint32_t interval_start_step,
        const Prepared (&)[NodeCapacity],
        std::uint16_t node_count,
        StepCount,
        TerminalAdjustment (&)[NodeCapacity]
    );
    __device__ static void apply_terminal_adjustment(
        const Prepared&,
        const TerminalAdjustment&,
        std::uint32_t step_count,
        State&
    );
    template<std::size_t NodeCapacity>
    __device__ static void draw_coupled(
        RandomContext&,
        const Prepared (&)[NodeCapacity],
        std::uint16_t node_count,
        Innovations (&)[NodeCapacity]
    );
    template<
        std::size_t NodeCapacity,
        typename IsActive,
        typename StepCount,
        typename StateAccessor>
    __device__ static void simulate_coupled_terminal(
        RandomContext&,
        const Prepared (&)[NodeCapacity],
        std::uint16_t node_count,
        std::uint32_t maximum_steps,
        IsActive,
        StepCount,
        StateAccessor
    );
    __device__ static void transition(
        const Prepared&, const Innovations&, const float*, State&
    );
    __device__ static float spot(const State&);
};

}  // namespace ai_factory::workbench::model::equity::bates::price_gradients
