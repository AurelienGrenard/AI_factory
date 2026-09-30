// Shared stopping-policy conventions for Longstaff-Schwartz sensitivities.
#pragma once

#include "common/longstaff_schwartz/exercise_decision.cuh"
#include "common/longstaff_schwartz/price_gradients/exercise_replay_strategy.hpp"
#include "common/longstaff_schwartz/regression_status.cuh"
#include "common/longstaff_schwartz/workspace.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

struct FrozenExerciseTimeReplay {};

template<typename RegressorT>
struct FrozenRegressionPolicyReplay {
    using Regressor = RegressorT;
};

template<typename Replay>
inline constexpr bool is_frozen_regression_policy_v = false;

template<typename Regressor>
inline constexpr bool is_frozen_regression_policy_v<
    FrozenRegressionPolicyReplay<Regressor>
> = true;

template<typename Regressor, typename Callable>
decltype(auto) dispatch_exercise_replay(
    ExerciseReplayStrategy strategy,
    Callable&& callable
) {
    switch (strategy) {
    case ExerciseReplayStrategy::frozen_exercise_time:
        return std::forward<Callable>(callable)
            .template operator()<FrozenExerciseTimeReplay>();
    case ExerciseReplayStrategy::frozen_regression_policy:
        return std::forward<Callable>(callable).template operator()<
            FrozenRegressionPolicyReplay<Regressor>
        >();
    }
    throw std::invalid_argument("Unknown exercise replay strategy.");
}

// One immutable continuation-function snapshot at one exercise date. The
// feature normalization is retained by the central PreparedRow; the snapshot
// therefore owns only the fitted coefficients and solve status.
template<typename Regressor>
struct FrozenRegressionSnapshot {
    double coefficients[Regressor::kBasisSize];
    RegressionStatus status;
};

template<typename Regressor>
inline StateFieldDescriptor frozen_regression_field_descriptor() {
    using Snapshot = FrozenRegressionSnapshot<Regressor>;
    return {sizeof(Snapshot), alignof(Snapshot)};
}

template<typename Regressor>
inline std::vector<StateFieldDescriptor> with_frozen_regression_field(
    std::vector<StateFieldDescriptor> descriptors = {}
) {
    descriptors.push_back(frozen_regression_field_descriptor<Regressor>());
    return descriptors;
}

template<typename Regressor>
__device__ __forceinline__ void capture_frozen_regression(
    FrozenRegressionSnapshot<Regressor>* snapshots,
    std::uint32_t regression_count,
    std::uint32_t backward_level,
    const double* coefficients,
    RegressionStatus status
) {
    const std::uint32_t exercise =
        regression_count - 1U - backward_level;
    auto& snapshot = snapshots[exercise];
    #pragma unroll
    for (std::size_t basis = 0U;
         basis < Regressor::kBasisSize;
         ++basis) {
        snapshot.coefficients[basis] = coefficients[basis];
    }
    snapshot.status = status;
}

// Product-independent forward replay of a fitted LSM stopping policy. The
// central row supplies the exact feature normalization used during fitting;
// the bumped row supplies the exercise payoff and contractual parameters.
template<typename PricingPolicy, typename Regressor>
struct FrozenRegressionExerciseHandler {
    using State = typename PricingPolicy::Dynamics::State;

    const typename PricingPolicy::PreparedRow& central;
    const typename PricingPolicy::PreparedRow& bumped;
    const FrozenRegressionSnapshot<Regressor>* snapshots;
    std::uint32_t exercise;

    __device__ __forceinline__ bool on_initial_state(const State&) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const State& state
    ) {
        if (observation >= central.regression_count) {
            exercise = central.regression_count;
            return false;
        }

        const auto& snapshot = snapshots[observation];
        if (snapshot.status != RegressionStatus::success) return true;

        const float immediate = PricingPolicy::replay_immediate_value(
            central, bumped, state, observation
        );
        if (!PricingPolicy::regression_candidate(immediate)) return true;

        const typename Regressor::Features features = Regressor::evaluate(
            PricingPolicy::replay_regression_input(central, state)
        );
        const double continuation = Regressor::predict(
            features, snapshot.coefficients
        );
        if (!exercise_is_preferred(immediate, continuation)) return true;

        exercise = observation;
        return false;
    }
};

static_assert(std::is_trivially_copyable_v<FrozenExerciseTimeReplay>);

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
