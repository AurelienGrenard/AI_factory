// Included device definitions of the Bates terminal QE-M and event-coupling adapter.
#pragma once

#include "model/equity/markovian/bates/price_gradients/coupled_dynamics.cuh"

#include "common/compound_poisson.cuh"
#include "common/monte_carlo/price_gradients/coupled_poisson_events.cuh"
#include "model/equity/markovian/bates/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::bates::price_gradients {

namespace detail {

using JumpInnovations = CoupledDynamics::TerminalAdjustment;

template<std::size_t NodeCapacity>
__device__ __forceinline__ void draw_coupled_jumps(
    philox::DomainRandomContext& random,
    std::uint32_t source_step,
    const CoupledDynamics::Prepared (&prepared)[NodeCapacity],
    const std::uint32_t (&step_counts)[NodeCapacity],
    std::uint8_t node_count,
    JumpInnovations (&innovations)[NodeCapacity]
) {
    float means[NodeCapacity]{};
    std::uint32_t counts[NodeCapacity]{};
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) {
            means[node] = prepared[node].poisson_mean
                * static_cast<float>(step_counts[node]);
            innovations[node].jump_standard_normal_sum = 0.0f;
        }
    }
    ::ai_factory::workbench::monte_carlo::price_gradients::
        replay_coupled_poisson_events<
            NodeCapacity,
            ::ai_factory::workbench::monte_carlo::price_gradients::
                StandardNormalEventMarks
        >(
            random, source_step, means, node_count, counts,
            [&](std::uint8_t node, float mark) {
                innovations[node].jump_standard_normal_sum += mark;
            }
        );
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) innovations[node].jump_count = counts[node];
    }
}

}  // namespace detail

__device__ __forceinline__ CoupledDynamics::Prepared CoupledDynamics::prepare(
    const ModelParameters& parameters,
    float dt
) {
    return bates::prepare_model(parameters, dt);
}

__device__ __forceinline__ CoupledDynamics::State CoupledDynamics::initial(
    const Prepared& prepared
) {
    return bates::initial_state(prepared);
}

__device__ __forceinline__ CoupledDynamics::ContinuousInnovations
CoupledDynamics::draw_continuous(
    RandomContext& random,
    const Prepared&
) {
    const auto step = random.next_step();
    auto continuous = random.source<
        compound_poisson::kContinuousSource
    >(step);
    philox::NormalPairCache continuous_cache;
    const float variance_normal = philox::next_normal(
        continuous, continuous_cache
    );
    const float stock_normal = philox::next_normal(
        continuous, continuous_cache
    );
    return {variance_normal, continuous.next(), stock_normal};
}

__device__ __forceinline__ void CoupledDynamics::transition_continuous(
    const Prepared& prepared,
    const ContinuousInnovations& innovations,
    State& state
) {
    heston::one_step_transition(
        prepared.heston,
        innovations.variance_normal,
        innovations.variance_uniform,
        innovations.stock_normal,
        state
    );
}

template<std::size_t NodeCapacity, typename StepCount>
__device__ __forceinline__ void CoupledDynamics::draw_terminal_adjustments(
    RandomContext& random,
    std::uint32_t interval_start_step,
    const Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    StepCount step_count,
    TerminalAdjustment (&adjustments)[NodeCapacity]
) {
    std::uint32_t step_counts[NodeCapacity]{};
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) step_counts[node] = step_count(node);
    }
    detail::draw_coupled_jumps(
        random,
        interval_start_step,
        prepared,
        step_counts,
        node_count,
        adjustments
    );
}

__device__ __forceinline__ void CoupledDynamics::apply_terminal_adjustment(
    const Prepared& prepared,
    const TerminalAdjustment& adjustment,
    std::uint32_t step_count,
    State& state
) {
    bates::apply_jump_interval(
        prepared,
        step_count,
        adjustment.jump_count,
        adjustment.jump_standard_normal_sum,
        state
    );
}

__device__ __forceinline__ CoupledDynamics::Innovations CoupledDynamics::draw(
    RandomContext& random,
    const Prepared& central
) {
    const auto step = random.next_step();
    auto continuous = random.source<
        compound_poisson::kContinuousSource
    >(step);
    philox::NormalPairCache continuous_cache;
    const float variance_normal = philox::next_normal(
        continuous, continuous_cache
    );
    const float stock_normal = philox::next_normal(
        continuous, continuous_cache
    );
    const float variance_uniform = continuous.next();
    const Prepared prepared[1U]{central};
    const std::uint32_t step_counts[1U]{1U};
    detail::JumpInnovations jump_innovations[1U]{};
    detail::draw_coupled_jumps(
        random, step, prepared, step_counts, 1U, jump_innovations
    );
    return {
        variance_normal,
        variance_uniform,
        stock_normal,
        jump_innovations[0U].jump_count,
        jump_innovations[0U].jump_standard_normal_sum,
    };
}

template<std::size_t NodeCapacity>
__device__ __forceinline__ void CoupledDynamics::draw_coupled(
    RandomContext& random,
    const Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    Innovations (&innovations)[NodeCapacity]
) {
    const auto step = random.next_step();
    auto continuous = random.source<
        compound_poisson::kContinuousSource
    >(step);
    philox::NormalPairCache continuous_cache;
    const float variance_normal = philox::next_normal(
        continuous, continuous_cache
    );
    const float stock_normal = philox::next_normal(
        continuous, continuous_cache
    );
    const float variance_uniform = continuous.next();
    std::uint32_t step_counts[NodeCapacity]{};
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) {
            step_counts[node] = 1U;
            innovations[node].variance_normal = variance_normal;
            innovations[node].variance_uniform = variance_uniform;
            innovations[node].stock_normal = stock_normal;
        }
    }
    detail::JumpInnovations jump_innovations[NodeCapacity]{};
    detail::draw_coupled_jumps(
        random, step, prepared, step_counts, node_count, jump_innovations
    );
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) {
            innovations[node].jump_count = jump_innovations[node].jump_count;
            innovations[node].jump_standard_normal_sum =
                jump_innovations[node].jump_standard_normal_sum;
        }
    }
}

template<
    std::size_t NodeCapacity,
    typename IsActive,
    typename StepCount,
    typename StateAccessor>
__device__ __forceinline__ void CoupledDynamics::simulate_coupled_terminal(
    RandomContext& random,
    const Prepared (&prepared)[NodeCapacity],
    std::uint8_t node_count,
    std::uint32_t maximum_steps,
    IsActive is_active,
    StepCount step_count,
    StateAccessor state
) {
    const auto interval_start_step = random.step_index;
    for (std::uint32_t step = 0U; step < maximum_steps; ++step) {
        const auto source_step = random.next_step();
        auto continuous = random.source<
            compound_poisson::kContinuousSource
        >(source_step);
        philox::NormalPairCache continuous_cache;
        const float variance_normal = philox::next_normal(
            continuous, continuous_cache
        );
        const float stock_normal = philox::next_normal(
            continuous, continuous_cache
        );
        const float variance_uniform = continuous.next();
        for (std::uint8_t node = 0U; node < node_count; ++node) {
            if (is_active(node) && step < step_count(node)) {
                heston::one_step_transition(
                    prepared[node].heston,
                    variance_normal,
                    variance_uniform,
                    stock_normal,
                    state(node)
                );
            }
        }
    }

    std::uint32_t step_counts[NodeCapacity]{};
    detail::JumpInnovations jump_innovations[NodeCapacity]{};
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        if (node < node_count) step_counts[node] = step_count(node);
    }
    detail::draw_coupled_jumps(
        random,
        interval_start_step,
        prepared,
        step_counts,
        node_count,
        jump_innovations
    );
    for (std::uint8_t node = 0U; node < node_count; ++node) {
        if (is_active(node)) {
            bates::apply_jump_interval(
                prepared[node],
                step_counts[node],
                jump_innovations[node].jump_count,
                jump_innovations[node].jump_standard_normal_sum,
                state(node)
            );
        }
    }
}

__device__ __forceinline__ void CoupledDynamics::transition(
    const Prepared& prepared,
    const Innovations& innovations,
    const float*,
    State& state
) {
    bates::one_step_transition(
        prepared,
        innovations.variance_normal,
        innovations.variance_uniform,
        innovations.stock_normal,
        innovations.jump_count,
        innovations.jump_standard_normal_sum,
        state
    );
}

__device__ __forceinline__ float CoupledDynamics::spot(const State& state) {
    return bates::DynamicsPolicy::spot(state);
}

}  // namespace ai_factory::workbench::model::equity::bates::price_gradients
