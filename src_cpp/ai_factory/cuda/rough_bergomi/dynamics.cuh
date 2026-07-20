#pragma once

#include "ai_factory/cuda/common/types.cuh"
#include "ai_factory/cuda/common/philox.cuh"

#include <cmath>
#include <cstddef>

namespace ai_factory::cuda::rough_bergomi_detail {

constexpr std::size_t kMaxRoughBergomiSteps = 272;

__device__ __forceinline__ double rough_bergomi_optimal_weight(
    double alpha,
    double dt,
    std::size_t k
) {
    const double kd = static_cast<double>(k);
    const double average =
        (pow(kd, alpha + 1.0) - pow(kd - 1.0, alpha + 1.0))
        / (alpha + 1.0);
    return pow(pow(average, 1.0 / alpha) * dt, alpha);
}

template <std::size_t MaxSteps>
__device__ __forceinline__ void prepare_rough_bergomi_path_normals(
    const RoughBergomiRow& row,
    std::size_t num_steps,
    std::size_t path,
    double sqrt_dt,
    double* dws
) {
    rng::NormalSequence normals(row.seed, 0U, path * num_steps);
    for (std::size_t step = 0; step < num_steps; ++step) {
        dws[step] = sqrt_dt * normals.next();
    }
}

template <std::size_t MaxSteps>
__device__ __forceinline__ double rough_bergomi_volterra_state(
    const RoughBergomiRow& row,
    std::size_t num_steps,
    std::size_t path,
    std::size_t step,
    double dt,
    double sqrt_dt,
    double singular_residual_normal,
    const double* weights,
    const double* dws
) {
    if (step == 0U) {
        return 0.0;
    }
    const double singular_covariance_scale =
        pow(dt, row.alpha + 0.5) / (row.alpha + 1.0);
    const double singular_residual_variance =
        pow(dt, 2.0 * row.alpha + 1.0)
        * (1.0 / (2.0 * row.alpha + 1.0)
           - 1.0 / ((row.alpha + 1.0) * (row.alpha + 1.0)));
    const double singular_residual_scale =
        sqrt(fmax(singular_residual_variance, 0.0));
    double y = singular_covariance_scale * (dws[step - 1U] / sqrt_dt)
               + singular_residual_scale * singular_residual_normal;
    for (std::size_t k = 2U; k <= step; ++k) {
        y += weights[k] * dws[step - k];
    }
    return y;
}

__device__ __forceinline__ double rough_bergomi_variance(
    const RoughBergomiRow& row,
    double y,
    double variance_time_power
) {
    const double variance_scale = sqrt(2.0 * row.alpha + 1.0);
    return row.forward_variance
           * exp(
               row.eta * variance_scale * y
               - 0.5 * row.eta * row.eta * variance_time_power
           );
}

template <std::size_t MaxSteps>
__device__ __forceinline__ double simulate_rough_bergomi_terminal_spot(
    const RoughBergomiRow& row,
    std::size_t num_steps,
    std::size_t path,
    const double* weights,
    const double* variance_time_powers
) {
    double dws[MaxSteps];
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift = (row.risk_free_rate - row.dividend_yield) * dt;
    const double rho_perp = sqrt(1.0 - row.rho * row.rho);
    prepare_rough_bergomi_path_normals<MaxSteps>(row, num_steps, path, sqrt_dt, dws);
    const auto first_index = path * num_steps;
    rng::NormalSequence residual_normals(row.seed, 1U, first_index);
    rng::NormalSequence stock_normals(row.seed, 2U, first_index);
    double log_spot = log(row.spot);
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double y = rough_bergomi_volterra_state<MaxSteps>(
            row, num_steps, path, step, dt, sqrt_dt,
            step == 0U ? 0.0 : residual_normals.next(), weights, dws
        );
        const double variance = rough_bergomi_variance(row, y, variance_time_powers[step]);
        const double dz = row.rho * dws[step] + rho_perp * sqrt_dt * stock_normals.next();
        log_spot += drift - 0.5 * variance * dt + sqrt(variance) * dz;
    }
    return exp(log_spot);
}

template <std::size_t MaxSteps>
__device__ __forceinline__ double simulate_rough_bergomi_max_spot(
    const RoughBergomiRow& row,
    std::size_t num_steps,
    std::size_t path,
    const double* weights,
    const double* variance_time_powers
) {
    double dws[MaxSteps];
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift = (row.risk_free_rate - row.dividend_yield) * dt;
    const double rho_perp = sqrt(1.0 - row.rho * row.rho);
    prepare_rough_bergomi_path_normals<MaxSteps>(
        row,
        num_steps,
        path,
        sqrt_dt,
        dws
    );
    const auto first_index = path * num_steps;
    rng::NormalSequence residual_normals(row.seed, 1U, first_index);
    rng::NormalSequence stock_normals(row.seed, 2U, first_index);

    double log_spot = log(row.spot);
    double max_spot = row.spot;
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double y = rough_bergomi_volterra_state<MaxSteps>(
            row,
            num_steps,
            path,
            step,
            dt,
            sqrt_dt,
            step == 0U ? 0.0 : residual_normals.next(),
            weights,
            dws
        );
        const double variance =
            rough_bergomi_variance(row, y, variance_time_powers[step]);
        const double dz =
            row.rho * dws[step]
            + rho_perp * sqrt_dt * stock_normals.next();
        log_spot += drift - 0.5 * variance * dt + sqrt(variance) * dz;
        max_spot = fmax(max_spot, exp(log_spot));
    }
    return max_spot;
}

template <std::size_t MaxSteps>
__device__ __forceinline__ double simulate_rough_bergomi_average_spot(
    const RoughBergomiRow& row,
    std::size_t num_steps,
    std::size_t path,
    const double* weights,
    const double* variance_time_powers
) {
    double dws[MaxSteps];
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift = (row.risk_free_rate - row.dividend_yield) * dt;
    const double rho_perp = sqrt(1.0 - row.rho * row.rho);
    prepare_rough_bergomi_path_normals<MaxSteps>(
        row,
        num_steps,
        path,
        sqrt_dt,
        dws
    );
    const auto first_index = path * num_steps;
    rng::NormalSequence residual_normals(row.seed, 1U, first_index);
    rng::NormalSequence stock_normals(row.seed, 2U, first_index);

    double log_spot = log(row.spot);
    double sum_spot = 0.0;
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double y = rough_bergomi_volterra_state<MaxSteps>(
            row,
            num_steps,
            path,
            step,
            dt,
            sqrt_dt,
            step == 0U ? 0.0 : residual_normals.next(),
            weights,
            dws
        );
        const double variance =
            rough_bergomi_variance(row, y, variance_time_powers[step]);
        const double dz =
            row.rho * dws[step]
            + rho_perp * sqrt_dt * stock_normals.next();
        log_spot += drift - 0.5 * variance * dt + sqrt(variance) * dz;
        sum_spot += exp(log_spot);
    }
    return sum_spot / static_cast<double>(num_steps);
}

template <std::size_t MaxSteps>
__device__ __forceinline__ double simulate_rough_bergomi_realized_volatility(
    const RoughBergomiRow& row,
    std::size_t num_steps,
    std::size_t path,
    const double* weights,
    const double* variance_time_powers,
    double observations_per_year
) {
    double dws[MaxSteps];
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift = (row.risk_free_rate - row.dividend_yield) * dt;
    const double rho_perp = sqrt(1.0 - row.rho * row.rho);
    prepare_rough_bergomi_path_normals<MaxSteps>(
        row,
        num_steps,
        path,
        sqrt_dt,
        dws
    );
    const auto first_index = path * num_steps;
    rng::NormalSequence residual_normals(row.seed, 1U, first_index);
    rng::NormalSequence stock_normals(row.seed, 2U, first_index);

    double sum_squared_log_returns = 0.0;
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double y = rough_bergomi_volterra_state<MaxSteps>(
            row,
            num_steps,
            path,
            step,
            dt,
            sqrt_dt,
            step == 0U ? 0.0 : residual_normals.next(),
            weights,
            dws
        );
        const double variance =
            rough_bergomi_variance(row, y, variance_time_powers[step]);
        const double dz =
            row.rho * dws[step]
            + rho_perp * sqrt_dt * stock_normals.next();
        const double log_return =
            drift - 0.5 * variance * dt + sqrt(variance) * dz;
        sum_squared_log_returns += log_return * log_return;
    }
    return sqrt(
        observations_per_year / static_cast<double>(num_steps)
        * sum_squared_log_returns
    );
}

template <std::size_t MaxSteps>
__device__ __forceinline__ void simulate_rough_bergomi_spot_path(
    const RoughBergomiRow& row,
    std::size_t num_steps,
    std::size_t path,
    const double* weights,
    const double* variance_time_powers,
    double* spot_paths
) {
    double dws[MaxSteps];
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift = (row.risk_free_rate - row.dividend_yield) * dt;
    const double rho_perp = sqrt(1.0 - row.rho * row.rho);
    const auto path_width = num_steps + 1U;
    prepare_rough_bergomi_path_normals<MaxSteps>(
        row,
        num_steps,
        path,
        sqrt_dt,
        dws
    );
    const auto first_index = path * num_steps;
    rng::NormalSequence residual_normals(row.seed, 1U, first_index);
    rng::NormalSequence stock_normals(row.seed, 2U, first_index);

    double log_spot = log(row.spot);
    spot_paths[path * path_width] = row.spot;
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double y = rough_bergomi_volterra_state<MaxSteps>(
            row,
            num_steps,
            path,
            step,
            dt,
            sqrt_dt,
            step == 0U ? 0.0 : residual_normals.next(),
            weights,
            dws
        );
        const double variance =
            rough_bergomi_variance(row, y, variance_time_powers[step]);
        const double dz =
            row.rho * dws[step]
            + rho_perp * sqrt_dt * stock_normals.next();
        log_spot += drift - 0.5 * variance * dt + sqrt(variance) * dz;
        spot_paths[path * path_width + step + 1U] = exp(log_spot);
    }
}

}  // namespace ai_factory::cuda::rough_bergomi_detail
