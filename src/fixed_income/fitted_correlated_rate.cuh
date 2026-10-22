// Curve fitting shared by correlated short-rate factors.
#pragma once
#include "philox/domains.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace ai_factory::workbench::fixed_income {
template<typename FactorPolicy,typename CurveProvider,typename Adapter>
struct FittedCorrelatedRatePolicy {
    using Parameters=typename Adapter::Parameters;
    using CurveParameters=typename CurveProvider::Parameters;
    struct Prepared {
        typename FactorPolicy::Prepared factor;
        Parameters model;
        CurveParameters curve;
        float dt;
    };
    struct State {
        typename FactorPolicy::State factor;
        float integral;
    };
    static constexpr unsigned kBrownianDimension=FactorPolicy::kBrownianDimension;
    static constexpr bool kUsesCurve=true;
    __host__ static bool valid_parameters(const Parameters& p) {
        return Adapter::valid_parameters(p);
    }
    __host__ static bool valid_curve(const CurveParameters& c) {
        return CurveProvider::valid_parameters(c);
    }
    __host__ static float internal_correlation(const Parameters& p)
        requires (kBrownianDimension==2U) {
        return Adapter::internal_correlation(p);
    }
    __device__ __forceinline__ static Prepared prepare(
        const Parameters& p,const CurveParameters& c,float dt) {
        return {FactorPolicy::prepare(Adapter::factor_parameters(p),
                    typename FactorPolicy::CurveParameters{},dt),p,c,dt};
    }
    __device__ __forceinline__ static State initial_state(const Prepared& p) {
        return {FactorPolicy::initial_state(p.factor),0.0f};
    }
    __device__ __forceinline__ static float one_step_transition(
        const Prepared& p,const float* z,philox::PhiloxKey key,
        std::uint64_t path,std::uint32_t step,std::uint32_t source,State& state) {
        const float factor_increment=FactorPolicy::one_step_transition(
            p.factor,z,key,path,step,source,state.factor);
        const float t0=static_cast<float>(step)*p.dt;
        // The deterministic shift is identical for every live path in a warp.
        const unsigned mask=__activemask();
        const int leader=__ffs(static_cast<int>(mask))-1;
        float shift=0.0f;
        if((threadIdx.x&31)==leader)
            shift=Adapter::template shift_integral<CurveProvider>(
                p.model,p.curve,t0,t0+p.dt);
        shift=__shfl_sync(mask,shift,leader);
        const float increment=factor_increment+shift;
        state.integral+=increment;
        return increment;
    }
};
}  // namespace ai_factory::workbench::fixed_income
