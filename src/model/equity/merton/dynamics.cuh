// Reusable CUDA interface for exact Merton jump-diffusion increments.
#pragma once

#include "common/philox.cuh"
#include "model/equity/merton/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::merton {

struct MertonPreparedParameters {
    float initial_log_spot;
    float drift_dt;
    float diffusion_std;
    float poisson_mean;
    float zero_jump_probability;
    float jump_log_mean;
    float jump_log_volatility;
};
struct MertonState {
    float log_spot;
};

struct MertonMeanPathResult {
    MertonState terminal_state;
    float arithmetic_mean;
};

struct MertonGeometricMeanPathResult {
    MertonState terminal_state;
    float geometric_mean;
};

struct MertonTwoTimePathResult {
    MertonState first_state;
    MertonState terminal_state;
};

struct MertonMaximumPathResult {
    MertonState terminal_state;
    float maximum_spot;
};

__device__ __forceinline__ MertonPreparedParameters prepare_model(
    const MertonModelParameters& parameters,
    float time_interval
);

__device__ __forceinline__ MertonPreparedParameters prepare_model(
    const MertonModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

__device__ __forceinline__ MertonState initial_state(
    const MertonPreparedParameters& model
);

__device__ __forceinline__ void one_step_transition(
    const MertonPreparedParameters& model,
    std::uint32_t jump_count,
    float diffusion_normal,
    float jump_normal,
    MertonState& state
);

__device__ __forceinline__ MertonState simulate_terminal_state(
    const MertonPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ MertonMeanPathResult simulate_mean_state(
    const MertonPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ MertonGeometricMeanPathResult
simulate_geometric_mean_state(
    const MertonPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ MertonTwoTimePathResult simulate_at_two_times(
    const MertonPreparedParameters& first_model,
    const MertonPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ MertonMaximumPathResult simulate_maximum_state(
    const MertonPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ MertonState simulate_on_regular_grid(
    const MertonPreparedParameters& initial_stub_model,
    const MertonPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* observed_spots
);

}  // namespace ai_factory::workbench::merton
