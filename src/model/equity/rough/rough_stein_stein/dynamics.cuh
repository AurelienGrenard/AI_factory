// Public path state and dynamics-policy declarations for rough Stein--Stein.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/volterra/fractional_resolvent_hybrid_kernel.cuh"
#include "model/equity/rough/rough_stein_stein/parameters.hpp"

#include <cuda_runtime.h>

#include <type_traits>

namespace ai_factory::workbench::model::equity::rough_stein_stein {

struct PreparedModel {
    float initial_log_spot;
    float volatility_level;
    float volatility_of_volatility;
    float time_step;
    float sqrt_time_step;
    float drift_time_step;
    float rho;
    float orthogonal_correlation;
};

struct State {
    float log_spot;
    float volatility;
};

struct PathPolicy {
    using Parameters = ModelParameters;
    using PreparedModel = rough_stein_stein::PreparedModel;
    using State = rough_stein_stein::State;

    static constexpr bool kNativeLogSpot = true;
    static constexpr bool kUsesVolterraVariance = false;

    __device__ __forceinline__ static
    volterra::FractionalResolventHybridKernelPolicy::Parameters
    kernel_parameters(const Parameters& parameters);
    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters,
        float time_step
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedModel& model
    );
    __device__ __forceinline__ static void advance(
        const PreparedModel& model,
        float volterra_value,
        float volterra_variance,
        float rough_normal,
        float independent_spot_normal,
        State& state
    );
    __device__ __forceinline__ static float spot(const State& state);
    __device__ __forceinline__ static float log_spot(const State& state);
};

static_assert(std::is_trivially_copyable_v<PreparedModel>);
static_assert(::ai_factory::workbench::equity::LogSpotStatePolicy<PathPolicy>);

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
