// Rough-SABR state mapping for a block-level Gaussian Volterra driver.
#pragma once

#include "common/equity/concepts.cuh"
#include "model/equity/rough/rough_sabr/parameters.hpp"

#include <cuda_runtime.h>

#include <type_traits>

namespace ai_factory::workbench::model::equity::rough_sabr {

struct PreparedModel {
    float initial_log_spot;
    float initial_volatility;
    float log_initial_volatility;
    float half_eta;
    float quarter_eta_squared;
    float beta;
    float one_minus_beta;
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

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters,
    float time_step
);

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
);

// The spot is evolved in the Lamperti coordinate S^(1-beta)/(1-beta),
// with the log-spot limit used at beta=1.
__device__ __forceinline__ void advance(
    const PreparedModel& prepared_model,
    float driver_value,
    float driver_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
);

struct PathPolicy {
    using Parameters = ModelParameters;
    using PreparedModel = rough_sabr::PreparedModel;
    using State = rough_sabr::State;

    static constexpr bool kNativeLogSpot = true;
    static constexpr bool kUsesDriverVariance = true;

    __device__ __forceinline__ static float driver_parameters(
        const Parameters& parameters
    );
    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters,
        float time_step
    );
    __device__ __forceinline__ static State initial_state(
        const PreparedModel& prepared_model
    );
    __device__ __forceinline__ static void advance(
        const PreparedModel& prepared_model,
        float driver_value,
        float driver_variance,
        float rough_normal,
        float independent_spot_normal,
        State& state
    );
    __device__ __forceinline__ static float spot(const State& state);
    __device__ __forceinline__ static float log_spot(const State& state);
};

static_assert(std::is_trivially_copyable_v<PreparedModel>);
static_assert(::ai_factory::workbench::equity::LogSpotStatePolicy<PathPolicy>);

}  // namespace ai_factory::workbench::model::equity::rough_sabr
