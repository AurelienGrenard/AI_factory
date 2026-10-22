// G2++ curve fitting around the correlated two-factor G2 process.
#pragma once
#include "fixed_income/fitted_correlated_rate.cuh"
#include "model/fixed_income/g2/correlated_dynamics.cuh"
#include "model/fixed_income/g2_plus_plus/fitted_analytics.cuh"
#include "model/fixed_income/g2/analytics_impl.cuh"

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus {
struct CorrelatedRateAdapter {
    using Parameters=ModelParameters;
    __host__ static bool valid_parameters(const Parameters& p) {
        return g2::valid_process_parameters(p.process);
    }
    __host__ static float internal_correlation(const Parameters& p) {
        return p.process.correlation;
    }
    __device__ __forceinline__ static g2::ModelParameters factor_parameters(
        const Parameters& p) { return {p.process,{0.0f,0.0f}}; }
    template<typename CurveProvider>
    __device__ __forceinline__ static float shift_integral(
        const Parameters& p,const typename CurveProvider::Parameters& c,
        float start,float end) {
        const fitted::FittedParameters<CurveProvider> fitted_p{p.process,c};
        return fitted::shift_integral(fitted_p,start,end);
    }
};
template<typename CurveProvider>
using CorrelatedRatePolicy=::ai_factory::workbench::fixed_income::FittedCorrelatedRatePolicy<
    g2::CorrelatedRatePolicy,CurveProvider,CorrelatedRateAdapter>;
}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus
