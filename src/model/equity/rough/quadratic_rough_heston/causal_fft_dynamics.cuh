// Causal FFT dynamics cell for the quadratic_rough_heston Volterra engine.
#pragma once

#include "volterra/fractional_causal_kernel.cuh"
#include "volterra/hybrid_fft.cuh"
#include "model/equity/rough/quadratic_rough_heston/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

struct CausalFftPathPolicy {
    using Parameters = ModelParameters;
    using KernelPolicy = volterra::FractionalCausalKernelPolicy;

    struct PreparedModel {
        float initial_log_spot;
        float initial_feedback;
        float quadratic_scale;
        float quadratic_shift;
        float variance_floor;
        float feedback_rate;
        float feedback_volatility;
        float drift_dt;
        float dt;
    };

    struct State {
        float log_spot;
        float feedback;
    };

    struct VolatilityState {
        float feedback;
    };

    static constexpr bool kNativeLogSpot = true;
    static constexpr unsigned int kBrownianDimension = 1U;

    __device__ __forceinline__ static float kernel_parameters(
        const Parameters& parameters
    ) { return parameters.hurst_exponent; }

    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters, float dt
    ) {
        return {
            logf(parameters.spot), parameters.initial_feedback,
            parameters.quadratic_scale, parameters.quadratic_shift,
            parameters.variance_floor, parameters.feedback_rate,
            parameters.feedback_volatility,
            (parameters.risk_free_rate - parameters.dividend_yield) * dt, dt,
        };
    }

    __device__ __forceinline__ static State initial_state(
        const PreparedModel& model
    ) { return {model.initial_log_spot, model.initial_feedback}; }

    __device__ __forceinline__ static VolatilityState initial_volatility_state(
        const PreparedModel& model
    ) { return {model.initial_feedback}; }

    // The same Brownian factor drives both the quadratic feedback and spot.
    __device__ __forceinline__ static float advance_volatility_cell(
        const PreparedModel& model,
        const KernelPolicy::PreparedKernel& kernel,
        float far_convolution,
        float normal,
        float,
        std::uint32_t step,
        VolatilityState& state,
        float& variance_at_step
    ) {
        const float feedback = step == 0U ? model.initial_feedback
            : model.initial_feedback + far_convolution;
        const float centered = feedback - model.quadratic_shift;
        const float variance = fmaf(
            model.quadratic_scale, centered * centered, model.variance_floor
        );
        const float root_variance = sqrtf(fmaxf(variance, 0.0f));
        variance_at_step = variance;
        state.feedback = feedback;
        const float raw_force = fmaf(
            model.feedback_rate * model.feedback_volatility
                / sqrtf(model.dt),
            root_variance * normal,
            -model.feedback_rate * feedback
        );
        const float balanced_force = raw_force / hypotf(
            1.0f, kernel.first_cell_integral * raw_force
        );
        return model.dt * balanced_force;
    }

    __device__ __forceinline__ static void advance_spot_cell(
        const PreparedModel& model,
        float variance_at_step,
        float spot_normal,
        float& log_spot
    ) {
        log_spot += model.drift_dt - 0.5f * variance_at_step * model.dt
            + sqrtf(fmaxf(variance_at_step, 0.0f))
                * sqrtf(model.dt) * spot_normal;
    }

    __device__ __forceinline__ static void advance_spot_cell_with_rate(
        const PreparedModel& model, float variance,
        float normal, float& log_spot, float carry_shift
    ) {
        auto adjusted = model;
        adjusted.drift_dt += carry_shift;
        advance_spot_cell(adjusted, variance, normal, log_spot);
    }

    // Legacy scalar interface.
    __device__ __forceinline__ static float advance_cell(
        const PreparedModel& model,
        const KernelPolicy::PreparedKernel& kernel,
        float far_convolution,
        philox::PhiloxKey key,
        std::uint64_t path,
        std::uint32_t step,
        State& state
    ) {
        const float normal = volterra::hybrid_fft::normal_at(key, path, step);
        VolatilityState volatility{state.feedback};
        float variance = 0.0f;
        const float source = advance_volatility_cell(
            model, kernel, far_convolution, normal, 0.0f,
            step, volatility, variance
        );
        advance_spot_cell(model, variance, normal, state.log_spot);
        state.feedback = volatility.feedback;
        return source;
    }

    __device__ __forceinline__ static float spot(const State& state) {
        return expf(state.log_spot);
    }
    __device__ __forceinline__ static float log_spot(const State& state) {
        return state.log_spot;
    }
};

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
