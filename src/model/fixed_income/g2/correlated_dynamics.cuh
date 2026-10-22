// Two-factor G2 short rate driven by two entries of the global correlation matrix.
#pragma once
#include "fixed_income/correlated_two_factor_gaussian_rate.cuh"
#include "fixed_income/no_curve.hpp"
#include "model/fixed_income/g2/parameter_domain.hpp"

namespace ai_factory::workbench::model::fixed_income::g2 {
struct CorrelatedRatePolicy {
    using Parameters=ModelParameters;
    using CurveParameters=::ai_factory::workbench::fixed_income::NoCurveParameters;
    using State=::ai_factory::workbench::fixed_income::CorrelatedTwoFactorGaussianRateState;
    using Prepared=::ai_factory::workbench::fixed_income::CorrelatedTwoFactorGaussianRateStep;
    static constexpr unsigned kBrownianDimension=2U;
    static constexpr bool kUsesCurve=false;
    __host__ static bool valid_parameters(const Parameters& p) {
        return g2::valid_parameters(p);
    }
    __host__ static float internal_correlation(const Parameters& p) {
        return p.process.correlation;
    }
    __device__ __forceinline__ static Prepared prepare(
        const Parameters& p,const CurveParameters&,float dt) {
        return ::ai_factory::workbench::fixed_income::prepare_correlated_two_factor_gaussian_rate(
            p.process.mean_reversion_x,p.process.volatility_x,p.initial_state.state_x,
            p.process.mean_reversion_y,p.process.volatility_y,p.initial_state.state_y,
            p.process.correlation,dt);
    }
    __device__ __forceinline__ static State initial_state(const Prepared& p) {
        return {p.x.initial_rate,p.y.initial_rate,0.0f};
    }
    __device__ __forceinline__ static float one_step_transition(
        const Prepared& p,const float* z,philox::PhiloxKey key,
        std::uint64_t path,std::uint32_t step,std::uint32_t source,State& state) {
        return ::ai_factory::workbench::fixed_income::advance_correlated_two_factor_gaussian_rate(
            p,z,key,path,step,source,state);
    }
};
}  // namespace ai_factory::workbench::model::fixed_income::g2
