// Reusable CUDA interface for Schobel-Zhu stochastic-volatility paths.
#pragma once

#include "common/philox.cuh"
#include "model/equity/schobel_zhu/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::schobel_zhu {

struct SchobelZhuPreparedParameters {
    float initial_log_spot;
    float initial_volatility;
    float long_run_volatility;
    float exp_mean_reversion_dt;
    float ou_std;
    float endpoint_increment_correlation;
    float endpoint_increment_residual;
    float drift_dt;
    float sqrt_dt;
    float correlation;
    float correlation_residual;
};
struct SchobelZhuState {
    float log_spot;
    float volatility;
};

struct SchobelZhuMeanPathResult {
    float arithmetic_mean;
};

struct SchobelZhuGeometricMeanPathResult {
    float geometric_mean;
};

struct SchobelZhuTwoTimePathResult {
    float first_spot;
    float terminal_spot;
};

struct SchobelZhuMaximumPathResult {
    float maximum_spot;
};

// ======================== Common equity dynamics =========================

__device__ __forceinline__ SchobelZhuPreparedParameters prepare_model(
    const SchobelZhuModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);

__device__ __forceinline__ SchobelZhuState initial_state(
    const SchobelZhuPreparedParameters& model
);

__device__ __forceinline__ void one_step_transition(
    const SchobelZhuPreparedParameters& model,
    float ou_normal,
    float increment_residual_normal,
    float asset_residual_normal,
    SchobelZhuState& state
);

__device__ __forceinline__ SchobelZhuState simulate_terminal_state(
    const SchobelZhuPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ SchobelZhuMeanPathResult simulate_mean_state(
    const SchobelZhuPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ SchobelZhuGeometricMeanPathResult
simulate_geometric_mean_state(
    const SchobelZhuPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ SchobelZhuTwoTimePathResult simulate_at_two_times(
    const SchobelZhuPreparedParameters& first_model,
    const SchobelZhuPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t first_num_steps,
    std::size_t second_num_steps
);

__device__ __forceinline__ SchobelZhuMaximumPathResult simulate_maximum_state(
    const SchobelZhuPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);

__device__ __forceinline__ SchobelZhuState simulate_on_regular_grid(
    const SchobelZhuPreparedParameters& initial_stub_model,
    const SchobelZhuPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* observed_spots,
    float* observed_volatilities
);

}  // namespace ai_factory::workbench::schobel_zhu
