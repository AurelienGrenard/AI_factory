// Causal FFT dynamics cell for the rough_heston Volterra engine.
#pragma once

#include "common/volterra/fractional_causal_kernel.cuh"
#include "common/volterra/hybrid_fft.cuh"
#include "model/equity/rough/rough_heston/parameters.hpp"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::equity::rough_heston {

struct CausalFftPathPolicy {
    using Parameters = ModelParameters;
    using KernelPolicy = volterra::FractionalCausalKernelPolicy;

    struct PreparedModel {
        float initial_log_spot;
        float initial_variance;
        float variance_drift;
        float mean_reversion;
        float volatility_of_variance;
        float rho;
        float orthogonal_correlation;
        float drift_dt;
        float dt;
    };

    struct State {
        float log_spot;
        float variance;
        float pending_singular_cell;
    };

    struct VolatilityState {
        float variance;
        float pending_singular_cell;
    };

    static constexpr bool kNativeLogSpot = true;
    static constexpr unsigned int kBrownianDimension = 2U;

    __device__ __forceinline__ static float kernel_parameters(
        const Parameters& parameters
    ) { return parameters.hurst_exponent; }

    __device__ __forceinline__ static PreparedModel prepare_model(
        const Parameters& parameters, float dt
    ) {
        return {
            logf(parameters.spot), parameters.initial_variance,
            parameters.variance_drift, parameters.mean_reversion,
            parameters.volatility_of_variance, parameters.rho,
            sqrtf(fmaxf(1.0f - parameters.rho * parameters.rho, 0.0f)),
            (parameters.risk_free_rate - parameters.dividend_yield) * dt, dt,
        };
    }

    __device__ __forceinline__ static State initial_state(
        const PreparedModel& model
    ) { return {model.initial_log_spot, model.initial_variance, 0.0f}; }

    __device__ __forceinline__ static VolatilityState initial_volatility_state(
        const PreparedModel& model
    ) { return {model.initial_variance, 0.0f}; }

    // Returns the causal source and the left-endpoint variance used by the
    // spot.  This operation has no dependency on the spot or the product.
    __device__ __forceinline__ static float advance_volatility_cell(
        const PreparedModel& model,
        const KernelPolicy::PreparedKernel& kernel,
        float far_convolution,
        float variance_normal,
        float singular_normal,
        std::uint32_t step,
        VolatilityState& state,
        float& variance_at_step
    ) {
        const float variance = step == 0U ? model.initial_variance
            : model.initial_variance + far_convolution
                + state.pending_singular_cell;
        const float root_variance = sqrtf(fmaxf(variance, 0.0f));
        variance_at_step = variance;
        state.variance = variance;
        state.pending_singular_cell = model.volatility_of_variance
            * root_variance * kernel.singular_independent_loading
            * singular_normal;
        return (model.variance_drift - model.mean_reversion * variance)
            * model.dt
            + model.volatility_of_variance * root_variance
                * (sqrtf(model.dt) * variance_normal);
    }

    __device__ __forceinline__ static void advance_spot_cell(
        const PreparedModel& model,
        float variance_at_step,
        float spot_normal,
        float& log_spot
    ) {
        const float positive_variance = fmaxf(variance_at_step, 0.0f);
        log_spot += model.drift_dt
            - 0.5f * positive_variance * model.dt
            + sqrtf(positive_variance) * sqrtf(model.dt) * spot_normal;
    }

    // Legacy scalar interface: its Philox mapping and marginal remain intact.
    __device__ __forceinline__ static float advance_cell(
        const PreparedModel& model,
        const KernelPolicy::PreparedKernel& kernel,
        float far_convolution,
        philox::PhiloxKey key,
        std::uint64_t path,
        std::uint32_t step,
        State& state
    ) {
        const float variance_normal = volterra::hybrid_fft::normal_at(
            key, path, 3ULL * step
        );
        const float singular_normal = volterra::hybrid_fft::normal_at(
            key, path, 3ULL * step + 1ULL
        );
        const float independent_spot_normal = volterra::hybrid_fft::normal_at(
            key, path, 3ULL * step + 2ULL
        );
        VolatilityState volatility{
            state.variance, state.pending_singular_cell
        };
        float variance = 0.0f;
        const float source = advance_volatility_cell(
            model, kernel, far_convolution, variance_normal,
            singular_normal, step, volatility, variance
        );
        const float spot_normal = model.rho * variance_normal
            + model.orthogonal_correlation * independent_spot_normal;
        advance_spot_cell(model, variance, spot_normal, state.log_spot);
        state.variance = volatility.variance;
        state.pending_singular_cell = volatility.pending_singular_cell;
        return source;
    }

    __device__ __forceinline__ static float spot(const State& state) {
        return expf(state.log_spot);
    }
    __device__ __forceinline__ static float log_spot(const State& state) {
        return state.log_spot;
    }
};

}  // namespace ai_factory::workbench::model::equity::rough_heston
