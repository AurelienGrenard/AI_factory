// Exact Gaussian short-rate and integral step conditional on its Brownian increment.
#pragma once

#include "common/philox_domains.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::fixed_income {

struct CorrelatedGaussianRateState {
    float short_rate;
    float integral;
};

struct CorrelatedGaussianRateStep {
    float decay;
    float mean_loading;
    float integral_state_loading;
    float integral_mean_loading;
    float state_brownian_loading;
    float state_bridge_loading;
    float integral_brownian_loading;
    float integral_bridge_loading;
    float integral_second_bridge_loading;
    float initial_rate;
};

// The shared normal is Delta W / sqrt(dt). Two independent bridge normals
// complete the exact joint Gaussian law of (r(t+dt), integral_t^{t+dt} r).
__device__ __forceinline__ CorrelatedGaussianRateStep
prepare_correlated_gaussian_rate(float mean_reversion, float long_run_mean,
                                 float volatility, float initial_rate,
                                 float dt) {
    const double a = static_cast<double>(mean_reversion);
    const double h = static_cast<double>(dt);
    const double sigma = static_cast<double>(volatility);
    double decay, b, state_z, state_bridge, integral_z;
    double integral_bridge, integral_second_bridge;
    if (a == 0.0) {
        decay = 1.0;
        b = h;
        state_z = sigma * sqrt(h);
        state_bridge = 0.0;
        integral_z = sigma * h * sqrt(h) * 0.5;
        integral_bridge = sigma * h * sqrt(h / 12.0);
        integral_second_bridge = 0.0;
    } else {
        const double one_minus_decay = -expm1(-a * h);
        const double one_minus_double_decay = -expm1(-2.0 * a * h);
        decay = 1.0 - one_minus_decay;
        b = one_minus_decay / a;
        const double x = a * h;
        state_z = sigma * b / sqrt(h);
        if (x < 0.01) {
            // Closed-form integral variances subtract nearly equal terms
            // when the time cell is fine. Integrate the centered Brownian
            // kernels with three-point Gauss-Legendre quadrature instead.
            const double mean_f = b / h;
            const double mean_g = 0.5 + x * (-1.0 / 6.0
                + x * (1.0 / 24.0 + x * (-1.0 / 120.0
                + x * (1.0 / 720.0 - x / 5040.0))));
            integral_z = sigma * h * sqrt(h) * mean_g;
            constexpr double node = 0.3872983346207416885;
            const double u[3] = {0.5 - node, 0.5, 0.5 + node};
            const double w[3] = {5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0};
            double ff = 0.0, fg = 0.0, gg = 0.0;
            for (int i = 0; i < 3; ++i) {
                const double f = exp(-x * u[i]) - mean_f;
                const double g = -expm1(-x * u[i]) / x - mean_g;
                ff += w[i] * f * f;
                fg += w[i] * f * g;
                gg += w[i] * g * g;
            }
            state_bridge = sigma * sqrt(h * fmax(0.0, ff));
            integral_bridge = state_bridge > 0.0
                ? sigma * sigma * h * h * fg / state_bridge : 0.0;
            integral_second_bridge = sqrt(fmax(0.0,
                sigma * sigma * h * h * h * gg
                    - integral_bridge * integral_bridge));
        } else {
            const double var_state = sigma * sigma
                * one_minus_double_decay / (2.0 * a);
            const double cov_state_integral = sigma * sigma / a
                * (b - one_minus_double_decay / (2.0 * a));
            const double var_integral = sigma * sigma / (a * a)
                * (h - 2.0 * b + one_minus_double_decay / (2.0 * a));
            integral_z = sigma * (h - b) / (a * sqrt(h));
            state_bridge = sqrt(fmax(0.0,
                var_state - state_z * state_z));
            integral_bridge = state_bridge > 0.0
                ? (cov_state_integral - state_z * integral_z)
                    / state_bridge : 0.0;
            integral_second_bridge = sqrt(fmax(0.0,
                var_integral - integral_z * integral_z
                    - integral_bridge * integral_bridge));
        }
    }
    return {
        static_cast<float>(decay),
        static_cast<float>(1.0 - decay),
        static_cast<float>(b),
        static_cast<float>(h - b),
        static_cast<float>(state_z),
        static_cast<float>(state_bridge),
        static_cast<float>(integral_z),
        static_cast<float>(integral_bridge),
        static_cast<float>(integral_second_bridge),
        initial_rate
    };
}

__device__ __forceinline__ float advance_correlated_gaussian_rate(
    const CorrelatedGaussianRateStep& prepared,
    float shared_brownian_normal,
    philox::PhiloxKey key, std::uint64_t path, std::uint32_t step,
    std::uint32_t auxiliary_source,
    float long_run_mean, CorrelatedGaussianRateState& state
) {
    const auto uniforms = philox::domain_uniform_quad(
        key, path, 0U, (auxiliary_source << 24U) | step);
    const auto bridge = philox::box_muller(
        uniforms.first, uniforms.second);
    const float state_noise = fmaf(
        prepared.state_brownian_loading, shared_brownian_normal,
        prepared.state_bridge_loading * bridge.first);
    const float integral_noise = fmaf(
        prepared.integral_brownian_loading, shared_brownian_normal,
        fmaf(prepared.integral_bridge_loading, bridge.first,
             prepared.integral_second_bridge_loading * bridge.second));
    const float previous = state.short_rate;
    const float increment = fmaf(prepared.integral_state_loading,
        previous, prepared.integral_mean_loading * long_run_mean)
        + integral_noise;
    state.short_rate = fmaf(prepared.decay, previous,
        prepared.mean_loading * long_run_mean + state_noise);
    state.integral += increment;
    return increment;
}

}  // namespace ai_factory::workbench::fixed_income
