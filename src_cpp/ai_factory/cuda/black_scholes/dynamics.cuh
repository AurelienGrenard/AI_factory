#pragma once

#include "ai_factory/cuda/common/philox.cuh"
#include "ai_factory/cuda/common/types.cuh"

#include <cmath>

namespace ai_factory::cuda::black_scholes_detail {

__device__ __forceinline__ double simulate_terminal_spot(
    const BlackScholesRow& row,
    std::size_t path,
    std::size_t
) {
    const double normal = rng::standard_normal(row.seed, 0U, path);
    return row.spot * exp(
        (row.risk_free_rate - row.dividend_yield
         - 0.5 * row.volatility * row.volatility) * row.maturity
        + row.volatility * sqrt(row.maturity) * normal
    );
}

__device__ inline double log_step(
    const BlackScholesRow& row,
    double dt,
    double sqrt_dt,
    double normal
) {
    return (row.risk_free_rate - row.dividend_yield - 0.5 * row.volatility * row.volatility)
               * dt
           + row.volatility * sqrt_dt * normal;
}

__device__ inline double simulate_max_spot(
    const BlackScholesRow& row,
    std::size_t path,
    std::size_t num_steps
) {
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift =
        (row.risk_free_rate - row.dividend_yield
         - 0.5 * row.volatility * row.volatility)
        * dt;
    const double diffusion = row.volatility * sqrt_dt;
    rng::NormalSequence normals(row.seed, 0U, path * num_steps);
    double spot = row.spot;
    double max_spot = row.spot;
    for (std::size_t step = 0; step < num_steps; ++step) {
        spot *= exp(fma(diffusion, normals.next(), drift));
        max_spot = fmax(max_spot, spot);
    }
    return max_spot;
}

__device__ inline double simulate_average_spot(
    const BlackScholesRow& row,
    std::size_t path,
    std::size_t num_steps
) {
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift =
        (row.risk_free_rate - row.dividend_yield
         - 0.5 * row.volatility * row.volatility)
        * dt;
    const double diffusion = row.volatility * sqrt_dt;
    rng::NormalSequence normals(row.seed, 0U, path * num_steps);
    double spot = row.spot;
    double sum = 0.0;
    for (std::size_t step = 0; step < num_steps; ++step) {
        spot *= exp(fma(diffusion, normals.next(), drift));
        sum += spot;
    }
    return sum / static_cast<double>(num_steps);
}

__device__ inline double simulate_realized_volatility(
    const BlackScholesRow& row,
    std::size_t path,
    std::size_t num_steps
) {
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift =
        (row.risk_free_rate - row.dividend_yield
         - 0.5 * row.volatility * row.volatility)
        * dt;
    const double diffusion = row.volatility * sqrt_dt;
    rng::NormalSequence normals(row.seed, 0U, path * num_steps);
    double sumsq = 0.0;
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double lr = fma(diffusion, normals.next(), drift);
        sumsq += lr * lr;
    }
    return sqrt(sumsq / row.maturity);
}

}  // namespace ai_factory::cuda::black_scholes_detail
