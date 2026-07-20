#pragma once

#include "ai_factory/cuda/common/barrier_pricing.cuh"
#include "ai_factory/cuda/common/philox.cuh"
#include "ai_factory/cuda/common/types.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace ai_factory::cuda::rough_heston_detail {

constexpr std::size_t kFactorCount = kRoughHestonFactorCount;

struct FactorCoefficients {
    double decay[kFactorCount];
    double drift[kFactorCount];
    double diffusion[kFactorCount];
};

__host__ __device__ inline FactorCoefficients make_factor_coefficients(
    double hurst,
    double maturity,
    double dt
) {
    FactorCoefficients result{};
    const double alpha = hurst + 0.5;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double measure_scale = sin(pi * alpha) / pi;
    const double lower_scale = 0.1 / maturity;
    const double upper_scale = 20.0 / dt;
    const double ratio = pow(
        upper_scale / lower_scale,
        1.0 / static_cast<double>(kFactorCount - 1U)
    );
    double left = 0.0;
    double right = lower_scale;
    for (std::size_t factor = 0; factor < kFactorCount; ++factor) {
        const double mass =
            measure_scale
            * (pow(right, 1.0 - alpha) - pow(left, 1.0 - alpha))
            / (1.0 - alpha);
        const double first_moment =
            measure_scale
            * (pow(right, 2.0 - alpha) - pow(left, 2.0 - alpha))
            / (2.0 - alpha);
        const double node = first_moment / mass;
        const double decay = exp(-node * dt);
        const double integrated_weight = mass * (-expm1(-node * dt)) / node;
        result.decay[factor] = decay;
        result.drift[factor] = integrated_weight;
        result.diffusion[factor] = integrated_weight / sqrt(dt);
        left = right;
        right *= ratio;
    }
    return result;
}

__device__ __forceinline__ void advance(
    const RoughHestonRow& row,
    const FactorCoefficients& coefficients,
    double dt,
    double variance_normal,
    double independent_normal,
    double* factors,
    double& log_spot,
    double& variance
) {
    const double positive_variance = fmax(variance, 0.0);
    const double root_variance = sqrt(positive_variance);
    const double common_drift = row.kappa * (row.theta - positive_variance);
    const double common_diffusion =
        row.volatility_of_variance * root_variance * variance_normal;
    double factor_sum = 0.0;
    #pragma unroll
    for (std::size_t factor = 0; factor < kFactorCount; ++factor) {
        factors[factor] =
            coefficients.decay[factor] * factors[factor]
            + coefficients.drift[factor] * common_drift
            + coefficients.diffusion[factor] * common_diffusion;
        factor_sum += factors[factor];
    }
    const double stock_normal =
        row.rho * variance_normal
        + sqrt(1.0 - row.rho * row.rho) * independent_normal;
    log_spot +=
        (row.risk_free_rate - row.dividend_yield - 0.5 * positive_variance) * dt
        + root_variance * sqrt(dt) * stock_normal;
    variance = fmax(row.initial_variance + factor_sum, 0.0);
}

template <typename Observer>
__device__ __forceinline__ double simulate(
    const RoughHestonRow& row,
    std::size_t path,
    std::size_t num_steps,
    Observer&& observer
) {
    const double dt = row.maturity / static_cast<double>(num_steps);
    const auto coefficients = make_factor_coefficients(
        row.hurst, row.maturity, dt
    );
    double factors[kFactorCount]{};
    double variance = row.initial_variance;
    double log_spot = log(row.spot);
    rng::NormalSequence variance_normals(row.seed, 0U, path * num_steps);
    rng::NormalSequence independent_normals(row.seed, 1U, path * num_steps);
    for (std::size_t step = 0; step < num_steps; ++step) {
        advance(
            row, coefficients, dt,
            variance_normals.next(), independent_normals.next(),
            factors, log_spot, variance
        );
        observer(step, exp(log_spot), variance);
    }
    return exp(log_spot);
}

template <typename Observer>
__device__ __forceinline__ double simulate_state_path(
    const RoughHestonRow& row,
    std::size_t path,
    std::size_t num_steps,
    Observer&& observer
) {
    const double dt = row.maturity / static_cast<double>(num_steps);
    const auto coefficients = make_factor_coefficients(
        row.hurst, row.maturity, dt
    );
    double factors[kFactorCount]{};
    double variance = row.initial_variance;
    double log_spot = log(row.spot);
    rng::NormalSequence variance_normals(row.seed, 0U, path * num_steps);
    rng::NormalSequence independent_normals(row.seed, 1U, path * num_steps);
    for (std::size_t step = 0; step < num_steps; ++step) {
        advance(
            row, coefficients, dt,
            variance_normals.next(), independent_normals.next(),
            factors, log_spot, variance
        );
        observer(step, exp(log_spot), variance, factors);
    }
    return exp(log_spot);
}

struct IgnoreObserver {
    __device__ __forceinline__ void operator()(
        std::size_t, double, double
    ) const {}
};

__device__ __forceinline__ double simulate_terminal_spot(
    const RoughHestonRow& row,
    std::size_t path,
    std::size_t num_steps
) {
    return simulate(row, path, num_steps, IgnoreObserver{});
}

struct BarrierSimulator {
    __device__ static barrier_detail::PathState run(
        const RoughHestonRow& row,
        std::size_t path,
        std::size_t num_steps,
        double barrier
    ) {
        bool hit = false;
        const bool up = barrier >= row.spot;
        struct Observer {
            double barrier;
            bool up;
            bool* hit;
            __device__ void operator()(std::size_t, double spot, double) const {
                *hit = *hit || (up ? spot >= barrier : spot <= barrier);
            }
        } observer{barrier, up, &hit};
        const double terminal = simulate(row, path, num_steps, observer);
        return {terminal, hit};
    }
};

}  // namespace ai_factory::cuda::rough_heston_detail
