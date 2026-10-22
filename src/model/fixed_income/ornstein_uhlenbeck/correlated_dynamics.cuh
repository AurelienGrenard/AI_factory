// OU short rate conditional on the shared Brownian increment.
#pragma once
#include "fixed_income/correlated_gaussian_rate.cuh"
#include "fixed_income/no_curve.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/parameter_domain.hpp"

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {
struct CorrelatedRatePolicy {
    using Parameters = ModelParameters;
    using CurveParameters = ::ai_factory::workbench::fixed_income::NoCurveParameters;
    using State = ::ai_factory::workbench::fixed_income::CorrelatedGaussianRateState;
    using Prepared = ::ai_factory::workbench::fixed_income::CorrelatedGaussianRateStep;
    static constexpr unsigned kBrownianDimension = 1U;
    static constexpr bool kUsesCurve = false;

    __host__ static bool valid_parameters(const Parameters& p) {
        return ornstein_uhlenbeck::valid_parameters(p);
    }
    __device__ __forceinline__ static Prepared prepare(
        const Parameters& p, const CurveParameters&, float dt) {
        return ::ai_factory::workbench::fixed_income::prepare_correlated_gaussian_rate(
            p.process.mean_reversion, 0.0f, p.process.volatility,
            p.initial_state, dt);
    }
    __device__ __forceinline__ static State initial_state(const Prepared& p) {
        return {p.initial_rate, 0.0f};
    }
    __device__ __forceinline__ static float one_step_transition(
        const Prepared& p, const float* z, philox::PhiloxKey key,
        std::uint64_t path, std::uint32_t step, std::uint32_t source,
        State& state) {
        return ::ai_factory::workbench::fixed_income::advance_correlated_gaussian_rate(
            p,z[0],key,path,step,source,0.0f,state);
    }
};
}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
