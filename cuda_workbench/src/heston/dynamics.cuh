// Device-side Heston dynamics implemented with Andersen's QE-M scheme.
// The file prepares row-constant coefficients once, advances one path step,
// and exposes a terminal-spot simulator reusable by terminal payoffs.
#pragma once

#include "common/philox.cuh"
#include "heston/common.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::heston {

// Scheme constants select the QE branch and split integrated variance weights.
constexpr float kQePsiCritical = 1.5f;
constexpr float kGamma1 = 0.5f;
constexpr float kGamma2 = 0.5f;

// Parameters below are used at every path step. Preparing them once per block
// removes repeated exp/division work from every simulated path.
struct HestonQeParameters {
    float initial_log_spot;
    float initial_variance;
    float theta;
    float exp_kdt;
    float variance_linear_scale;
    float variance_constant_scale;
    float drift_dt;
    float k0;
    float k1;
    float k2;
    float k3;
    float k4;
    float martingale_a;
    std::uint64_t seed;
};

// HestonState is the minimal evolving state kept private to one CUDA thread.
struct HestonState {
    float log_spot;
    float variance;
};

// Precompute all row- and time-step-dependent QE-M coefficients once per block.
__device__ __forceinline__ HestonQeParameters prepare_model(
    const HestonModelInput& input,
    float maturity,
    std::size_t num_steps,
    std::uint64_t seed
) {
    // Short aliases keep the coefficient formulas close to their notation.
    const float kappa = input.kappa;
    const float theta = input.theta;
    const float gamma = input.gamma;
    const float rho = input.rho;
    const float dt = maturity / static_cast<float>(num_steps);
    const float exp_kdt = expf(-kappa * dt);
    const float one_minus_exp = 1.0f - exp_kdt;
    const float gamma2 = gamma * gamma;
    const float drift_dt =
        (input.risk_free_rate - input.dividend_yield) * dt;
    const float kappa_rho_over_gamma = kappa * rho / gamma;
    const float rho_over_gamma = rho / gamma;
    const float k2 =
        kGamma2 * dt * (kappa_rho_over_gamma - 0.5f) + rho_over_gamma;
    const float k4 = kGamma2 * dt * (1.0f - rho * rho);

    return {
        logf(input.spot),
        input.initial_variance,
        theta,
        exp_kdt,
        gamma2 * exp_kdt * one_minus_exp / kappa,
        theta * gamma2 * one_minus_exp * one_minus_exp / (2.0f * kappa),
        drift_dt,
        drift_dt - rho * kappa * theta * dt / gamma,
        kGamma1 * dt * (kappa_rho_over_gamma - 0.5f) - rho_over_gamma,
        k2,
        kGamma1 * dt * (1.0f - rho * rho),
        k4,
        k2 + 0.5f * k4,
        seed,
    };
}

// Construct the t=0 log-spot and variance for one simulated path.
__device__ __forceinline__ HestonState initial_state(
    const HestonQeParameters& model
) {
    return {model.initial_log_spot, model.initial_variance};
}

// One Andersen QE-M transition. The first branch approximates the conditional
// CIR variance with a quadratic Gaussian law; the second uses a mass at zero
// plus an exponential tail. The martingale correction controls stock drift.
__device__ __forceinline__ void one_step_qe_martingale_transition(
    const HestonQeParameters& model,
    float variance_normal,
    float variance_uniform,
    float stock_normal,
    HestonState& state
) {
    const float previous_variance = fmaxf(state.variance, 0.0f);
    const float conditional_mean =
        model.theta
        + (previous_variance - model.theta) * model.exp_kdt;
    const float conditional_variance =
        previous_variance * model.variance_linear_scale
        + model.variance_constant_scale;

    float next_variance = 0.0f;
    float log_moment = 0.0f;
    bool martingale_valid = true;

    if (conditional_mean > 0.0f && conditional_variance > 0.0f) {
        const float psi = conditional_variance
                        / (conditional_mean * conditional_mean);
        if (psi <= kQePsiCritical) {
            const float inverse_psi = 1.0f / psi;
            const float root_term =
                fmaxf(2.0f * inverse_psi - 1.0f, 0.0f);
            const float b2 =
                root_term
                + sqrtf(2.0f * inverse_psi) * sqrtf(root_term);
            const float b = sqrtf(fmaxf(b2, 0.0f));
            const float a = conditional_mean / (1.0f + b2);
            const float shifted = b + variance_normal;
            next_variance = a * shifted * shifted;

            const float denominator =
                1.0f - 2.0f * model.martingale_a * a;
            if (denominator > 0.0f) {
                log_moment = model.martingale_a * b2 * a / denominator
                             - 0.5f * logf(denominator);
            } else {
                martingale_valid = false;
            }
        } else {
            const float probability_zero = (psi - 1.0f) / (psi + 1.0f);
            const float beta =
                (1.0f - probability_zero) / conditional_mean;
            next_variance = variance_uniform <= probability_zero
                ? 0.0f
                : logf((1.0f - probability_zero) / (1.0f - variance_uniform))
                      / beta;

            if (model.martingale_a < beta) {
                const float moment =
                    probability_zero
                    + beta * (1.0f - probability_zero)
                          / (beta - model.martingale_a);
                martingale_valid = moment > 0.0f;
                if (martingale_valid) log_moment = logf(moment);
            } else {
                martingale_valid = false;
            }
        }
    }

    const float variance_integral_proxy = fmaxf(
        model.k3 * previous_variance + model.k4 * next_variance,
        0.0f
    );
    if (martingale_valid) {
        state.log_spot +=
            model.drift_dt - log_moment
            - 0.5f * model.k3 * previous_variance
            + model.k2 * next_variance
            + sqrtf(variance_integral_proxy) * stock_normal;
    } else {
        // This fallback is the non-martingale-corrected QE log update. It is
        // used only if the moment required by the correction does not exist.
        state.log_spot +=
            model.k0
            + model.k1 * previous_variance
            + model.k2 * next_variance
            + sqrtf(variance_integral_proxy) * stock_normal;
    }
    state.variance = next_variance;
}

// Simulate one complete path and return only its terminal spot.
__device__ __forceinline__ float simulate_terminal_spot(
    const HestonQeParameters& model,
    std::size_t path,
    std::size_t num_steps
) {
    HestonState state = initial_state(model);
    const std::uint64_t first_index =
        static_cast<std::uint64_t>(path) * static_cast<std::uint64_t>(num_steps);

    // Independent stream numbers separate variance normals, stock normals,
    // and variance uniforms while the path index selects disjoint ranges.
    rng::NormalSequence variance_normals(model.seed, 0ULL, first_index);
    rng::NormalSequence stock_normals(model.seed, 1ULL, first_index);
    rng::UniformSequence variance_uniforms(model.seed, 2ULL, first_index);
    for (std::size_t step = 0; step < num_steps; ++step) {
        one_step_qe_martingale_transition(
            model,
            variance_normals.next(),
            variance_uniforms.next(),
            stock_normals.next(),
            state
        );
    }
    return expf(state.log_spot);
}

}  // namespace ai_factory::workbench::heston
