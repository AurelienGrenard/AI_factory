// Reusable CUDA interface for flat-xi0 rough-Bergomi hybrid paths.
#pragma once

#include "common/philox.cuh"
#include "model/equity/rough_bergomi/dataset.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::rough_bergomi {

// Scalar coefficients shared by all paths of one model/product row.
struct RoughBergomiPreparedParameters {
    float initial_log_spot;
    float initial_variance;
    float log_initial_variance;
    float eta;
    float half_eta_squared;
    float alpha_plus_one;
    float two_h;
    float sqrt_two_h;
    float time_step;
    float sqrt_time_step;
    float time_step_to_alpha;
    float drift_time_step;
    float rho;
    float orthogonal_correlation;
    float singular_driver_loading;
    float singular_independent_loading;
};

// Evolving values that remain in registers for one Monte Carlo path.
struct RoughBergomiState {
    float log_spot;
    float variance;
};

// One global-memory lane. Consecutive threads own consecutive values at a date.
struct RoughBergomiHistoryView {
    float* brownian_increments;
    std::size_t stride;
};

// Deterministic arrays prepared cooperatively in block shared memory.
struct RoughBergomiHybridGridView {
    const float* far_weights;
    const float* log_variance_corrections;
};

struct RoughBergomiMeanPathResult {
    float arithmetic_mean;
};

struct RoughBergomiGeometricMeanPathResult {
    float geometric_mean;
};

struct RoughBergomiTwoTimePathResult {
    float first_spot;
    float terminal_spot;
};

struct RoughBergomiMaximumPathResult {
    float maximum_spot;
};

// Prepare the regular time step and the kappa=1 Gaussian coupling.
__device__ __forceinline__ RoughBergomiPreparedParameters prepare_model(
    const RoughBergomiModelParameters& parameters,
    float maturity,
    std::uint32_t num_steps
);

// Fill the two caller-owned shared arrays cooperatively; synchronize afterwards.
__device__ __forceinline__ void prepare_hybrid_grid(
    const RoughBergomiPreparedParameters& model,
    std::uint32_t step_count,
    float* far_weights,
    float* log_variance_corrections,
    unsigned int thread_index,
    unsigned int thread_count
);

// Construct the time-zero spot and flat forward variance state.
__device__ __forceinline__ RoughBergomiState initial_state(
    const RoughBergomiPreparedParameters& model
);

// Apply one deterministic hybrid transition from three independent normals.
__device__ __forceinline__ void one_step_transition(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    std::uint32_t step_index,
    float rough_driver_normal,
    float singular_independent_normal,
    float spot_independent_normal,
    RoughBergomiState& state
);

// Draw one step from the single path-local scalar uniform sequence.
__device__ __forceinline__ void simulate_one_step(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    std::uint32_t step_index,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    RoughBergomiState& state
);

__device__ __forceinline__ RoughBergomiState simulate_terminal_state(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

__device__ __forceinline__ RoughBergomiMeanPathResult simulate_mean_state(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

__device__ __forceinline__ RoughBergomiGeometricMeanPathResult
simulate_geometric_mean_state(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

// Observe two indices without restarting the non-Markovian history.
__device__ __forceinline__ RoughBergomiTwoTimePathResult
simulate_at_two_times(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t first_step_index,
    std::uint32_t terminal_step_index
);

__device__ __forceinline__ RoughBergomiMaximumPathResult
simulate_maximum_state(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);

// Store observation_count - 1 spots and return the maturity state directly.
__device__ __forceinline__ RoughBergomiState simulate_on_regular_grid(
    const RoughBergomiPreparedParameters& model,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count,
    std::size_t path_count,
    float* __restrict__ observed_spots
);

}  // namespace ai_factory::workbench::rough_bergomi
