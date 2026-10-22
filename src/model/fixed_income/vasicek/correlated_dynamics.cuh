// Vasicek rate factor for a cross-asset Brownian vector.
#pragma once

#include "fixed_income/correlated_gaussian_rate.cuh"
#include "fixed_income/no_curve.hpp"
#include "model/fixed_income/vasicek/parameters.hpp"

#include <cmath>

namespace ai_factory::workbench::model::fixed_income::vasicek {

struct CorrelatedRatePolicy {
    using Parameters = ModelParameters;
    using CurveParameters = ::ai_factory::workbench::fixed_income::NoCurveParameters;
    using State = ::ai_factory::workbench::fixed_income::CorrelatedGaussianRateState;
    struct Prepared {
        ::ai_factory::workbench::fixed_income::CorrelatedGaussianRateStep transition;
        float long_run_mean;
    };
    static constexpr unsigned int kBrownianDimension = 1U;
    static constexpr bool kUsesCurve = false;

    __host__ static bool valid_parameters(const Parameters& p) {
        return std::isfinite(p.process.mean_reversion)
            && p.process.mean_reversion >= 0.0f
            && std::isfinite(p.process.long_term_mean)
            && std::isfinite(p.process.volatility)
            && p.process.volatility >= 0.0f
            && std::isfinite(p.initial_state);
    }

    __device__ __forceinline__ static Prepared prepare(
        const Parameters& parameters, const CurveParameters&, float dt
    ) {
        return {
            ::ai_factory::workbench::fixed_income::prepare_correlated_gaussian_rate(
                parameters.process.mean_reversion,
                parameters.process.long_term_mean,
                parameters.process.volatility,
                parameters.initial_state, dt),
            parameters.process.long_term_mean
        };
    }

    __device__ __forceinline__ static State initial_state(
        const Prepared& prepared
    ) {
        return {prepared.transition.initial_rate, 0.0f};
    }

    __device__ __forceinline__ static float one_step_transition(
        const Prepared& prepared, const float* brownian,
        philox::PhiloxKey key, std::uint64_t path, std::uint32_t step,
        std::uint32_t auxiliary_source, State& state
    ) {
        return ::ai_factory::workbench::fixed_income::advance_correlated_gaussian_rate(
            prepared.transition, brownian[0], key, path, step,
            auxiliary_source, prepared.long_run_mean, state);
    }
};

}  // namespace ai_factory::workbench::model::fixed_income::vasicek
