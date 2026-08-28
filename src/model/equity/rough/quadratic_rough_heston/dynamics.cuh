// Public state, N-factor preparation and dynamics-policy declarations for quadratic rough Heston.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "common/volterra/exponential_kernel.cuh"
#include "model/equity/rough/quadratic_rough_heston/parameters.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

template<std::size_t FactorCount>
requires (FactorCount > 0U)
struct State {
    float log_spot;
    float feedback_factors[FactorCount];
};

template<std::size_t FactorCount>
requires (FactorCount > 0U)
struct PreparedDynamics {
    volterra::ExponentialKernel<FactorCount> kernel;
    float initial_log_spot;
    float initial_feedback;
    float quadratic_scale;
    float quadratic_shift;
    float variance_floor;
    float factor_decay[FactorCount];
    float factor_drift_integral[FactorCount];
    float feedback_cell_loading;
    float feedback_rate;
    float feedback_volatility;
    float drift_dt;
    float dt;
    float sqrt_dt;
    float inverse_sqrt_dt;
};

template<std::size_t FactorCount>
struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics =
        quadratic_rough_heston::PreparedDynamics<FactorCount>;
    using RandomContext = philox::NormalRandomContext;
    using State = quadratic_rough_heston::State<FactorCount>;

    static constexpr bool kNativeLogSpot = true;
    static constexpr bool kPartitionInvariantAdvance = true;

    __device__ __forceinline__ static State initial_state(
        const PreparedDynamics& dynamics
    );
    __device__ __forceinline__ static void simulate_one_step(
        const PreparedDynamics& dynamics,
        RandomContext& random,
        State& state
    );
    __device__ __forceinline__ static void advance(
        const PreparedDynamics& dynamics,
        std::uint32_t step_count,
        RandomContext& random,
        State& state
    );
    __device__ __forceinline__ static float spot(const State& state);
    __device__ __forceinline__ static float log_spot(const State& state);
};

static_assert(std::is_trivially_copyable_v<PreparedDynamics<7U>>);
static_assert(simulation::PreparedFixedStepDynamicsPolicy<DynamicsPolicy<7U>>);
static_assert(::ai_factory::workbench::equity::LogSpotDynamicsPolicy<
    DynamicsPolicy<7U>
>);

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
