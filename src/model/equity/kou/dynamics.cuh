// Reusable CUDA interface for exact Kou jump-diffusion increments.
#pragma once

#include "common/philox.cuh"
#include "model/equity/kou/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::kou {

struct KouPreparedParameters {
    float initial_log_spot;
    float drift_dt;
    float diffusion_std;
    float poisson_mean;
    float zero_jump_probability;
    float up_probability;
    float inverse_positive_jump_rate;
    float inverse_negative_jump_rate;
};
struct KouState {
    float log_spot;
};

struct KouMeanPathResult {
    KouState terminal_state;
    float arithmetic_mean;
};

struct KouGeometricMeanPathResult {
    KouState terminal_state;
    float geometric_mean;
};

struct KouTwoTimePathResult {
    KouState first_state;
    KouState terminal_state;
};

struct KouMaximumPathResult {
    KouState terminal_state;
    float maximum_spot;
};

__device__ __forceinline__ KouPreparedParameters prepare_model(
    const KouModelParameters& parameters,
    float time_interval
);

__device__ __forceinline__ KouPreparedParameters prepare_model(
    const KouModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

__device__ __forceinline__ KouState initial_state(
    const KouPreparedParameters& model
);

__device__ __forceinline__ void one_step_transition(
    const KouPreparedParameters& model,
    float diffusion_normal,
    float jump_log_sum,
    KouState& state
);

__device__ __forceinline__ KouState simulate_terminal_state(
    const KouPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ KouMeanPathResult simulate_mean_state(
    const KouPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ KouGeometricMeanPathResult
simulate_geometric_mean_state(
    const KouPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ KouTwoTimePathResult simulate_at_two_times(
    const KouPreparedParameters& first_model,
    const KouPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
);

__device__ __forceinline__ KouMaximumPathResult simulate_maximum_state(
    const KouPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ KouState simulate_on_regular_grid(
    const KouPreparedParameters& initial_stub_model,
    const KouPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* observed_spots
);

}  // namespace ai_factory::workbench::kou
