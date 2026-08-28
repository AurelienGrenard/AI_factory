// Public rough-Bergomi state-mapping contract used by the Gaussian Volterra kernel.
#pragma once

#include "common/equity/concepts.cuh"
#include "model/equity/rough/rough_bergomi/parameters.hpp"

#include <cuda_runtime.h>

#include <type_traits>

namespace ai_factory::workbench::model::equity::rough_bergomi {

struct PreparedModel {
    float initial_log_spot;
    float initial_variance;
    float log_initial_variance;
    float eta;
    float half_eta_squared;
    float time_step;
    float sqrt_time_step;
    float drift_time_step;
    float rho;
    float orthogonal_correlation;
};

struct State {
    float log_spot;
    float variance;
};

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters,
    float time_step
);

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
);

// Freeze v at the left endpoint, advance log(S), then map the Volterra kernel
// at the right endpoint to the next variance.
__device__ __forceinline__ void advance(
    const PreparedModel& prepared_model,
    float volterra_value,
    float volterra_variance,
    float rough_normal,
    float independent_spot_normal,
    State& state
);

struct PathPolicy {
    using Parameters = ModelParameters;
    using PreparedModel = rough_bergomi::PreparedModel;
    using State = rough_bergomi::State;

    static constexpr bool kNativeLogSpot = true;
    static constexpr bool kUsesVolterraVariance = true;

    __device__ __forceinline__ static float kernel_parameters(
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

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
