// Hull-White curve fitting around the correlated OU factor.
#pragma once
#include "common/fixed_income/fitted_correlated_rate.cuh"
#include "model/fixed_income/hull_white/fitted_analytics.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/analytics_impl.cuh"
#include "model/fixed_income/hull_white/parameter_domain.hpp"
#include "model/fixed_income/ornstein_uhlenbeck/correlated_dynamics.cuh"

namespace ai_factory::workbench::model::fixed_income::hull_white {
struct CorrelatedRateAdapter {
    using Parameters=ModelParameters;
    __host__ static bool valid_parameters(const Parameters& p) {
        return hull_white::valid_parameters(p);
    }
    __device__ __forceinline__ static ornstein_uhlenbeck::ModelParameters
    factor_parameters(const Parameters& p) {
        return {{p.mean_reversion,p.volatility},0.0f};
    }
    template<typename CurveProvider>
    __device__ __forceinline__ static float shift_integral(
        const Parameters& p,const typename CurveProvider::Parameters& c,
        float start,float end) {
        const fitted::FittedParameters<CurveProvider> fitted_p{
            {p.mean_reversion,p.volatility},c};
        return fitted::shift_integral(fitted_p,start,end);
    }
};
template<typename CurveProvider>
using CorrelatedRatePolicy=::ai_factory::workbench::fixed_income::FittedCorrelatedRatePolicy<
    ornstein_uhlenbeck::CorrelatedRatePolicy,CurveProvider,CorrelatedRateAdapter>;
}  // namespace ai_factory::workbench::model::fixed_income::hull_white
