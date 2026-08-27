// Hull-White stochastic-factor adapters over the shared OU dynamics.
#pragma once

#include "common/simulation/adapted_dynamics.cuh"
#include "model/fixed_income/hull_white/parameters.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/dynamics.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::hull_white {

using BaseDynamics =
    model::fixed_income::ornstein_uhlenbeck::DynamicsPolicy;

struct ProcessParameterAdapter {
    __device__ __forceinline__ static BaseDynamics::Parameters adapt(
        const ModelParameters& parameters
    ) {
        return {
            {parameters.mean_reversion, parameters.volatility},
            0.0f,
        };
    }
};

using DynamicsPolicy = simulation::AdaptedExactTransitionDynamicsPolicy<
    ModelParameters,
    BaseDynamics,
    ProcessParameterAdapter
>;

namespace joint {

using BaseDynamics =
    model::fixed_income::ornstein_uhlenbeck::joint::DynamicsPolicy;

struct JointProcessParameterAdapter {
    __device__ __forceinline__ static BaseDynamics::Parameters adapt(
        const ModelParameters& parameters
    ) {
        return {
            {parameters.mean_reversion, parameters.volatility},
            0.0f,
        };
    }
};

using DynamicsPolicy = simulation::AdaptedExactTransitionDynamicsPolicy<
    ModelParameters,
    BaseDynamics,
    JointProcessParameterAdapter
>;

}  // namespace joint
}  // namespace ai_factory::workbench::model::fixed_income::hull_white
