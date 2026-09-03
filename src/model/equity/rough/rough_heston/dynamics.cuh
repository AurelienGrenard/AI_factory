// Fixed-factor Markovian lift and weak rough-Heston CUDA contract.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/philox.cuh"
#include "common/volterra/exponential_kernel.cuh"
#include "model/equity/rough/rough_heston/parameters.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::model::equity::rough_heston {

template<std::size_t FactorCount>
requires (FactorCount > 0U)
struct State {
    float log_spot;
    float variance_factors[FactorCount];
};

// Host-prepared invariants for one dt. The (FactorCount+1)-dimensional matrix
// exponential is reduced to the factor matrix and affine shift stored here.
template<std::size_t FactorCount>
requires (FactorCount > 0U)
struct PreparedDynamics {
    volterra::ExponentialKernel<FactorCount> kernel;
    float initial_log_spot;
    float initial_factors[FactorCount];
    float ode_half_step[FactorCount * FactorCount];
    float ode_half_shift[FactorCount];
    float orthogonal_variance_scale;
    float orthogonal_drift_scale;
    float weight_sum;
    float weak_variance_scale;
    float correlated_constant;
    float half_dt_first_node;
    float correlated_weight_scale;
    float rho_over_volatility;
    float spot_drift_dt;
};

template<std::size_t FactorCount>
struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = rough_heston::PreparedDynamics<FactorCount>;
    using RandomContext = philox::NormalRandomContext;
    using State = rough_heston::State<FactorCount>;

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
static_assert(simulation::PreparedFixedStepDynamicsPolicy<DynamicsPolicy<2U>>);
static_assert(!simulation::FixedStepDynamicsPolicy<DynamicsPolicy<2U>>);
static_assert(::ai_factory::workbench::equity::LogSpotDynamicsPolicy<
    DynamicsPolicy<7U>
>);

}  // namespace ai_factory::workbench::model::equity::rough_heston
