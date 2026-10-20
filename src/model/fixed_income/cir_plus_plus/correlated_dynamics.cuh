// CIR++ curve fitting around the Brownian-driven CIR factor.
#pragma once
#include "common/fixed_income/fitted_correlated_rate.cuh"
#include "model/fixed_income/cir/correlated_dynamics.cuh"
#include "model/fixed_income/cir_plus_plus/fitted_analytics_impl.cuh"

namespace ai_factory::workbench::model::fixed_income::cir_plus_plus {
struct CorrelatedRateAdapter {
    using Parameters=ModelParameters;
    __host__ static bool valid_parameters(const Parameters& p) {
        return cir::valid_parameters(p);
    }
    __device__ __forceinline__ static cir::ModelParameters factor_parameters(
        const Parameters& p) { return p; }
    template<typename CurveProvider>
    __device__ __forceinline__ static float shift_integral(
        const Parameters& p,const typename CurveProvider::Parameters& c,
        float start,float end) {
        const fitted::FittedParameters<CurveProvider> fitted_p{p,c};
        return fitted::shift_integral(fitted_p,start,end);
    }
};
template<typename CurveProvider>
using CorrelatedRatePolicy=::ai_factory::workbench::fixed_income::FittedCorrelatedRatePolicy<
    cir::CorrelatedRatePolicy,CurveProvider,CorrelatedRateAdapter>;
}  // namespace ai_factory::workbench::model::fixed_income::cir_plus_plus
