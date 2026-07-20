#pragma once

#include "ai_factory/cuda/common/types.cuh"
#include "ai_factory/cuda/common/philox.cuh"

#include <cmath>

namespace ai_factory::cuda::heston_detail {

using rng::NormalPair;
using rng::RandomQuad;
using rng::standard_normal;
using rng::standard_normal_pair;
using rng::standard_normal_quad;
using rng::standard_uniform;
using rng::standard_uniform_quad;

constexpr int kHestonEulerFullTruncation = 0;
constexpr int kHestonAndersenQeMartingale = 2;
constexpr double kQePsiCritical = 1.5;
constexpr double kGamma1 = 0.5;
constexpr double kGamma2 = 0.5;

struct QeStep {
    double next_variance;
    double log_moment;
    bool martingale_valid;
};

struct QeCoefficients {
    double exp_kdt;
    double variance_linear_scale;
    double variance_constant_scale;
    double drift_dt;
    double k0;
    double k1;
    double k2;
    double k3;
    double k4;
    double martingale_a;
};

struct BarrierState {
    double terminal_spot;
    bool hit;
};

__device__ __forceinline__ QeCoefficients make_qe_coefficients(
    const HestonRow& row,
    double dt
);

__device__ __forceinline__ void advance_heston_step(
    const HestonRow& row, double dt, double sqrt_dt, double drift_scale,
    double correlation_scale, double kappa_dt, double vol_of_var_sqrt_dt,
    double spot_shock, double independent_variance_shock,
    double& spot, double& variance
);

__device__ __forceinline__ void advance_heston_qe_step_precomputed(
    const HestonRow& row,
    const QeCoefficients& coefficients,
    double variance_normal,
    double variance_uniform,
    double stock_normal,
    double& log_spot,
    double& variance
);

__device__ __forceinline__ double simulate_heston_terminal_spot(
    const HestonRow& row,
    std::size_t path,
    std::size_t num_steps
) {
    const double dt = row.maturity / static_cast<double>(num_steps);
    if (row.scheme == kHestonEulerFullTruncation) {
        const double sqrt_dt = sqrt(dt);
        double spot = row.spot;
        double variance = row.initial_variance;
        rng::NormalSequence spot_normals(row.seed, 0U, path * num_steps);
        rng::NormalSequence variance_normals(row.seed, 1U, path * num_steps);
        for (std::size_t step = 0; step < num_steps; ++step) {
            advance_heston_step(
                row, dt, sqrt_dt, row.risk_free_rate - row.dividend_yield,
                sqrt(1.0 - row.rho * row.rho), row.kappa * dt,
                row.volatility_of_variance * sqrt_dt,
                spot_normals.next(), variance_normals.next(), spot, variance
            );
        }
        return spot;
    }
    double log_spot = log(row.spot);
    double variance = row.initial_variance;
    const auto coefficients = make_qe_coefficients(row, dt);
    rng::NormalSequence variance_normals(row.seed, 0U, path * num_steps);
    rng::NormalSequence stock_normals(row.seed, 1U, path * num_steps);
    rng::UniformSequence variance_uniforms(row.seed, 2U, path * num_steps);
    for (std::size_t step = 0; step < num_steps; ++step) {
        advance_heston_qe_step_precomputed(
            row,
            coefficients,
            variance_normals.next(),
            variance_uniforms.next(),
            stock_normals.next(),
            log_spot,
            variance
        );
    }
    return exp(log_spot);
}
__device__ __forceinline__ void advance_heston_step(
    const HestonRow& row,
    double dt,
    double sqrt_dt,
    double drift_scale,
    double correlation_scale,
    double kappa_dt,
    double vol_of_var_sqrt_dt,
    double spot_shock,
    double independent_variance_shock,
    double& spot,
    double& variance
) {
    const double variance_floor = fmax(variance, 0.0);
    const double sqrt_variance_floor = sqrt(variance_floor);
    const double variance_shock =
        row.rho * spot_shock + correlation_scale * independent_variance_shock;
    spot *= exp(
        (drift_scale - 0.5 * variance_floor) * dt
        + sqrt_variance_floor * sqrt_dt * spot_shock
    );
    variance += kappa_dt * (row.theta - variance_floor);
    variance += vol_of_var_sqrt_dt * sqrt_variance_floor * variance_shock;
    variance = fmax(variance, 0.0);
}

__device__ __forceinline__ QeStep advance_qe_variance(
    const HestonRow& row,
    double variance,
    double dt,
    double variance_normal,
    double variance_uniform,
    double martingale_a
) {
    const double exp_kdt = exp(-row.kappa * dt);
    const double one_minus_exp = 1.0 - exp_kdt;
    const double m = row.theta + (variance - row.theta) * exp_kdt;
    const double xi2 = row.volatility_of_variance * row.volatility_of_variance;
    const double s2 =
        variance * xi2 * exp_kdt * one_minus_exp / row.kappa
        + row.theta * xi2 * one_minus_exp * one_minus_exp / (2.0 * row.kappa);

    if (m <= 0.0 || s2 <= 0.0) {
        return {0.0, 0.0, true};
    }

    const double psi = s2 / (m * m);
    if (psi <= kQePsiCritical) {
        const double inv_psi = 1.0 / psi;
        const double b2 =
            2.0 * inv_psi - 1.0
            + sqrt(2.0 * inv_psi) * sqrt(fmax(2.0 * inv_psi - 1.0, 0.0));
        const double b = sqrt(fmax(b2, 0.0));
        const double a = m / (1.0 + b2);
        const double shifted = b + variance_normal;
        const double next_variance = a * shifted * shifted;
        const double denominator = 1.0 - 2.0 * martingale_a * a;
        if (denominator <= 0.0) {
            return {next_variance, 0.0, false};
        }
        const double log_moment =
            martingale_a * b2 * a / denominator - 0.5 * log(denominator);
        return {next_variance, log_moment, true};
    }

    const double p = (psi - 1.0) / (psi + 1.0);
    const double beta = (1.0 - p) / m;
    const double next_variance =
        variance_uniform <= p
            ? 0.0
            : log((1.0 - p) / (1.0 - variance_uniform)) / beta;
    if (martingale_a >= beta) {
        return {next_variance, 0.0, false};
    }
    const double moment = p + beta * (1.0 - p) / (beta - martingale_a);
    return {next_variance, log(moment), moment > 0.0};
}

__device__ __forceinline__ void advance_heston_qe_step(
    const HestonRow& row,
    double dt,
    double variance_normal,
    double variance_uniform,
    double stock_normal,
    double& log_spot,
    double& variance
) {
    const double rho = row.rho;
    const double xi = row.volatility_of_variance;
    const double drift_dt = (row.risk_free_rate - row.dividend_yield) * dt;
    const double kappa_rho_over_xi = row.kappa * rho / xi;
    const double rho_over_xi = rho / xi;
    const double k1 = kGamma1 * dt * (kappa_rho_over_xi - 0.5) - rho_over_xi;
    const double k2 = kGamma2 * dt * (kappa_rho_over_xi - 0.5) + rho_over_xi;
    const double k3 = kGamma1 * dt * (1.0 - rho * rho);
    const double k4 = kGamma2 * dt * (1.0 - rho * rho);
    const double martingale_a = k2 + 0.5 * k4;
    const double previous_variance = fmax(variance, 0.0);
    const auto qe = advance_qe_variance(
        row,
        previous_variance,
        dt,
        variance_normal,
        variance_uniform,
        martingale_a
    );
    const double next_variance = qe.next_variance;
    const double variance_integral_proxy =
        fmax(k3 * previous_variance + k4 * next_variance, 0.0);

    if (row.scheme == kHestonAndersenQeMartingale && qe.martingale_valid) {
        log_spot += drift_dt - qe.log_moment - 0.5 * k3 * previous_variance
                    + k2 * next_variance
                    + sqrt(variance_integral_proxy) * stock_normal;
    } else {
        const double k0 = drift_dt - rho * row.kappa * row.theta * dt / xi;
        log_spot += k0 + k1 * previous_variance + k2 * next_variance
                    + sqrt(variance_integral_proxy) * stock_normal;
    }
    variance = next_variance;
}

__device__ __forceinline__ QeCoefficients make_qe_coefficients(
    const HestonRow& row,
    double dt
) {
    const double xi = row.volatility_of_variance;
    const double rho = row.rho;
    const double exp_kdt = exp(-row.kappa * dt);
    const double one_minus_exp = 1.0 - exp_kdt;
    const double xi2 = xi * xi;
    const double drift_dt = (row.risk_free_rate - row.dividend_yield) * dt;
    const double kappa_rho_over_xi = row.kappa * rho / xi;
    const double rho_over_xi = rho / xi;
    const double k2 =
        kGamma2 * dt * (kappa_rho_over_xi - 0.5) + rho_over_xi;
    const double k4 = kGamma2 * dt * (1.0 - rho * rho);
    return {
        exp_kdt,
        xi2 * exp_kdt * one_minus_exp / row.kappa,
        row.theta * xi2 * one_minus_exp * one_minus_exp / (2.0 * row.kappa),
        drift_dt,
        drift_dt - rho * row.kappa * row.theta * dt / xi,
        kGamma1 * dt * (kappa_rho_over_xi - 0.5) - rho_over_xi,
        k2,
        kGamma1 * dt * (1.0 - rho * rho),
        k4,
        k2 + 0.5 * k4,
    };
}

__device__ __forceinline__ void advance_heston_qe_step_precomputed(
    const HestonRow& row,
    const QeCoefficients& coefficients,
    double variance_normal,
    double variance_uniform,
    double stock_normal,
    double& log_spot,
    double& variance
) {
    const double previous_variance = fmax(variance, 0.0);
    const double m =
        row.theta + (previous_variance - row.theta) * coefficients.exp_kdt;
    const double s2 = previous_variance * coefficients.variance_linear_scale
                      + coefficients.variance_constant_scale;

    double next_variance = 0.0;
    double log_moment = 0.0;
    bool martingale_valid = true;
    if (m > 0.0 && s2 > 0.0) {
        const double psi = s2 / (m * m);
        if (psi <= kQePsiCritical) {
            const double inv_psi = 1.0 / psi;
            const double b2 =
                2.0 * inv_psi - 1.0
                + sqrt(2.0 * inv_psi)
                      * sqrt(fmax(2.0 * inv_psi - 1.0, 0.0));
            const double b = sqrt(fmax(b2, 0.0));
            const double a = m / (1.0 + b2);
            const double shifted = b + variance_normal;
            next_variance = a * shifted * shifted;
            const double denominator = 1.0 - 2.0 * coefficients.martingale_a * a;
            if (denominator > 0.0) {
                log_moment =
                    coefficients.martingale_a * b2 * a / denominator
                    - 0.5 * log(denominator);
            } else {
                martingale_valid = false;
            }
        } else {
            const double p = (psi - 1.0) / (psi + 1.0);
            const double beta = (1.0 - p) / m;
            next_variance =
                variance_uniform <= p
                    ? 0.0
                    : log((1.0 - p) / (1.0 - variance_uniform)) / beta;
            if (coefficients.martingale_a < beta) {
                const double moment =
                    p + beta * (1.0 - p) / (beta - coefficients.martingale_a);
                log_moment = log(moment);
                martingale_valid = moment > 0.0;
            } else {
                martingale_valid = false;
            }
        }
    }

    const double variance_integral_proxy = fmax(
        coefficients.k3 * previous_variance + coefficients.k4 * next_variance,
        0.0
    );
    if (row.scheme == kHestonAndersenQeMartingale && martingale_valid) {
        log_spot += coefficients.drift_dt - log_moment
                    - 0.5 * coefficients.k3 * previous_variance
                    + coefficients.k2 * next_variance
                    + sqrt(variance_integral_proxy) * stock_normal;
    } else {
        log_spot += coefficients.k0 + coefficients.k1 * previous_variance
                    + coefficients.k2 * next_variance
                    + sqrt(variance_integral_proxy) * stock_normal;
    }
    variance = next_variance;
}

__device__ __forceinline__ void advance_heston_qe_max_step(
    const HestonRow& row,
    const QeCoefficients& coefficients,
    double variance_normal,
    double variance_uniform,
    double stock_normal,
    double& log_spot,
    double& variance,
    double& max_spot
) {
    advance_heston_qe_step_precomputed(
        row,
        coefficients,
        variance_normal,
        variance_uniform,
        stock_normal,
        log_spot,
        variance
    );
    max_spot = fmax(max_spot, exp(log_spot));
}

__device__ __forceinline__ void advance_heston_qe_sum_step(
    const HestonRow& row,
    const QeCoefficients& coefficients,
    double variance_normal,
    double variance_uniform,
    double stock_normal,
    double& log_spot,
    double& variance,
    double& sum_spot
) {
    advance_heston_qe_step_precomputed(
        row,
        coefficients,
        variance_normal,
        variance_uniform,
        stock_normal,
        log_spot,
        variance
    );
    sum_spot += exp(log_spot);
}

__device__ __forceinline__ double simulate_heston_qe_max_spot(
    const HestonRow& row,
    std::size_t num_steps,
    std::size_t path,
    double dt
) {
    const auto coefficients = make_qe_coefficients(row, dt);
    double log_spot = log(row.spot);
    double max_spot = row.spot;
    double variance = row.initial_variance;
    std::size_t step = 0U;
    std::uint64_t normal_index = path * num_steps;

    while (step < num_steps && (normal_index & 3ULL) != 0ULL) {
        advance_heston_qe_max_step(
            row,
            coefficients,
            standard_normal(row.seed, 0U, normal_index),
            standard_uniform(row.seed, 2U, normal_index),
            standard_normal(row.seed, 1U, normal_index),
            log_spot,
            variance,
            max_spot
        );
        ++step;
        ++normal_index;
    }

    for (; step + 3U < num_steps; step += 4U, normal_index += 4ULL) {
        const auto block_index = normal_index / 4ULL;
        const auto variance_normals =
            standard_normal_quad(row.seed, 0U, block_index);
        const auto variance_uniforms =
            standard_uniform_quad(row.seed, 2U, block_index);
        const auto stock_normals =
            standard_normal_quad(row.seed, 1U, block_index);

        advance_heston_qe_max_step(
            row,
            coefficients,
            variance_normals.first,
            variance_uniforms.first,
            stock_normals.first,
            log_spot,
            variance,
            max_spot
        );
        advance_heston_qe_max_step(
            row,
            coefficients,
            variance_normals.second,
            variance_uniforms.second,
            stock_normals.second,
            log_spot,
            variance,
            max_spot
        );
        advance_heston_qe_max_step(
            row,
            coefficients,
            variance_normals.third,
            variance_uniforms.third,
            stock_normals.third,
            log_spot,
            variance,
            max_spot
        );
        advance_heston_qe_max_step(
            row,
            coefficients,
            variance_normals.fourth,
            variance_uniforms.fourth,
            stock_normals.fourth,
            log_spot,
            variance,
            max_spot
        );
    }

    for (; step < num_steps; ++step, ++normal_index) {
        advance_heston_qe_max_step(
            row,
            coefficients,
            standard_normal(row.seed, 0U, normal_index),
            standard_uniform(row.seed, 2U, normal_index),
            standard_normal(row.seed, 1U, normal_index),
            log_spot,
            variance,
            max_spot
        );
    }
    return max_spot;
}

__device__ __forceinline__ double simulate_heston_qe_average_spot(
    const HestonRow& row,
    std::size_t num_steps,
    std::size_t path,
    double dt
) {
    const auto coefficients = make_qe_coefficients(row, dt);
    double log_spot = log(row.spot);
    double variance = row.initial_variance;
    double sum_spot = 0.0;
    std::size_t step = 0U;
    std::uint64_t normal_index = path * num_steps;

    while (step < num_steps && (normal_index & 3ULL) != 0ULL) {
        advance_heston_qe_sum_step(
            row,
            coefficients,
            standard_normal(row.seed, 0U, normal_index),
            standard_uniform(row.seed, 2U, normal_index),
            standard_normal(row.seed, 1U, normal_index),
            log_spot,
            variance,
            sum_spot
        );
        ++step;
        ++normal_index;
    }

    for (; step + 3U < num_steps; step += 4U, normal_index += 4ULL) {
        const auto block_index = normal_index / 4ULL;
        const auto variance_normals =
            standard_normal_quad(row.seed, 0U, block_index);
        const auto variance_uniforms =
            standard_uniform_quad(row.seed, 2U, block_index);
        const auto stock_normals =
            standard_normal_quad(row.seed, 1U, block_index);

        advance_heston_qe_sum_step(
            row,
            coefficients,
            variance_normals.first,
            variance_uniforms.first,
            stock_normals.first,
            log_spot,
            variance,
            sum_spot
        );
        advance_heston_qe_sum_step(
            row,
            coefficients,
            variance_normals.second,
            variance_uniforms.second,
            stock_normals.second,
            log_spot,
            variance,
            sum_spot
        );
        advance_heston_qe_sum_step(
            row,
            coefficients,
            variance_normals.third,
            variance_uniforms.third,
            stock_normals.third,
            log_spot,
            variance,
            sum_spot
        );
        advance_heston_qe_sum_step(
            row,
            coefficients,
            variance_normals.fourth,
            variance_uniforms.fourth,
            stock_normals.fourth,
            log_spot,
            variance,
            sum_spot
        );
    }

    for (; step < num_steps; ++step, ++normal_index) {
        advance_heston_qe_sum_step(
            row,
            coefficients,
            standard_normal(row.seed, 0U, normal_index),
            standard_uniform(row.seed, 2U, normal_index),
            standard_normal(row.seed, 1U, normal_index),
            log_spot,
            variance,
            sum_spot
        );
    }
    return sum_spot / static_cast<double>(num_steps);
}

template <bool Up>
__device__ __forceinline__ void advance_heston_qe_barrier_step(
    const HestonRow& row,
    const QeCoefficients& coefficients,
    double variance_normal,
    double variance_uniform,
    double stock_normal,
    double barrier,
    double& log_spot,
    double& variance,
    bool& hit
) {
    advance_heston_qe_step_precomputed(
        row,
        coefficients,
        variance_normal,
        variance_uniform,
        stock_normal,
        log_spot,
        variance
    );
    const double spot = exp(log_spot);
    hit = hit || (Up ? spot >= barrier : spot <= barrier);
}

template <bool Up>
__device__ __forceinline__ BarrierState simulate_heston_qe_barrier(
    const HestonRow& row,
    std::size_t num_steps,
    std::size_t path,
    double barrier
) {
    const double dt = row.maturity / static_cast<double>(num_steps);
    const auto coefficients = make_qe_coefficients(row, dt);
    double log_spot = log(row.spot);
    double variance = row.initial_variance;
    bool hit = false;
    std::size_t step = 0U;
    std::uint64_t index = path * num_steps;
    while (step < num_steps && (index & 3ULL) != 0ULL) {
        advance_heston_qe_barrier_step<Up>(
            row,
            coefficients,
            standard_normal(row.seed, 0U, index),
            standard_uniform(row.seed, 2U, index),
            standard_normal(row.seed, 1U, index),
            barrier,
            log_spot,
            variance,
            hit
        );
        ++step;
        ++index;
    }
    for (; step + 3U < num_steps; step += 4U, index += 4ULL) {
        const auto block = index / 4ULL;
        const auto zv = standard_normal_quad(row.seed, 0U, block);
        const auto uv = standard_uniform_quad(row.seed, 2U, block);
        const auto zs = standard_normal_quad(row.seed, 1U, block);
        advance_heston_qe_barrier_step<Up>(row, coefficients, zv.first, uv.first, zs.first, barrier, log_spot, variance, hit);
        advance_heston_qe_barrier_step<Up>(row, coefficients, zv.second, uv.second, zs.second, barrier, log_spot, variance, hit);
        advance_heston_qe_barrier_step<Up>(row, coefficients, zv.third, uv.third, zs.third, barrier, log_spot, variance, hit);
        advance_heston_qe_barrier_step<Up>(row, coefficients, zv.fourth, uv.fourth, zs.fourth, barrier, log_spot, variance, hit);
    }
    for (; step < num_steps; ++step, ++index) {
        advance_heston_qe_barrier_step<Up>(
            row,
            coefficients,
            standard_normal(row.seed, 0U, index),
            standard_uniform(row.seed, 2U, index),
            standard_normal(row.seed, 1U, index),
            barrier,
            log_spot,
            variance,
            hit
        );
    }
    return {exp(log_spot), hit};
}


}  // namespace ai_factory::cuda::heston_detail
