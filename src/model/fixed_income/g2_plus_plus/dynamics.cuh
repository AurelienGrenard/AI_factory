// G2++ stochastic-factor adapters over the shared G2 dynamics.
#pragma once

#include "common/simulation/adapted_dynamics.cuh"
#include "model/fixed_income/g2/dynamics.cuh"
#include "model/fixed_income/g2_plus_plus/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus {

using BaseDynamics = model::fixed_income::g2::DynamicsPolicy;

struct ProcessParameterAdapter {
    __device__ __forceinline__ static BaseDynamics::Parameters adapt(
        const ModelParameters& parameters
    ) {
        return {parameters.process, {0.0f, 0.0f}};
    }
};

using DynamicsPolicy = simulation::AdaptedExactTransitionDynamicsPolicy<
    ModelParameters,
    BaseDynamics,
    ProcessParameterAdapter
>;

namespace joint {

using BaseDynamics = model::fixed_income::g2::joint::DynamicsPolicy;

struct JointProcessParameterAdapter {
    __device__ __forceinline__ static BaseDynamics::Parameters adapt(
        const ModelParameters& parameters
    ) {
        return {parameters.process, {0.0f, 0.0f}};
    }
};

using DynamicsPolicy = simulation::AdaptedExactTransitionDynamicsPolicy<
    ModelParameters,
    BaseDynamics,
    JointProcessParameterAdapter
>;

}  // namespace joint
}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus
