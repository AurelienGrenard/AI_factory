// Reusable CUDA interface for absorbed Milstein CEV paths.
#pragma once

#include "common/philox.cuh"
#include "model/equity/cev/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::cev {

struct CevPreparedParameters {
    float initial_spot;
    float drift_dt;
    float diffusion_scale;
    float milstein_scale;
    float beta;
};
struct CevState {
    float spot;
};

struct CevMeanPathResult {
    CevState terminal_state;
    float arithmetic_mean;
};

struct CevGeometricMeanPathResult {
    CevState terminal_state;
    float geometric_mean;
};

struct CevTwoTimePathResult {
    CevState first_state;
    CevState terminal_state;
};

struct CevMaximumPathResult {
    CevState terminal_state;
    float maximum_spot;
};

__device__ __forceinline__ CevPreparedParameters prepare_model(
    const CevModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

__device__ __forceinline__ CevState initial_state(
    const CevPreparedParameters& model
);

__device__ __forceinline__ void one_step_transition(
    const CevPreparedParameters& model,
    float normal,
    CevState& state
);

__device__ __forceinline__ CevState simulate_terminal_state(
    const CevPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ CevMeanPathResult simulate_mean_state(
    const CevPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ CevGeometricMeanPathResult
simulate_geometric_mean_state(
    const CevPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ CevTwoTimePathResult simulate_at_two_times(
    const CevPreparedParameters& first_model,
    const CevPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t first_num_steps,
    std::size_t second_num_steps
);

__device__ __forceinline__ CevMaximumPathResult simulate_maximum_state(
    const CevPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ CevState simulate_on_regular_grid(
    const CevPreparedParameters& initial_stub_model,
    const CevPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* observed_spots
);

}  // namespace ai_factory::workbench::cev
