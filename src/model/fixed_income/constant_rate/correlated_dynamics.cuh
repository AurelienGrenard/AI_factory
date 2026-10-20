// Zero-factor short-rate policy: same path contract as stochastic rates.
#pragma once

#include "common/fixed_income/no_curve.hpp"
#include "common/philox.cuh"
#include "model/fixed_income/constant_rate/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::constant_rate {

struct CorrelatedRatePolicy {
    using Parameters = ModelParameters;
    using CurveParameters = ::ai_factory::workbench::fixed_income::NoCurveParameters;
    struct Prepared { float rate_dt; };
    struct State { float integral; };
    static constexpr unsigned kBrownianDimension = 0U;
    static constexpr bool kUsesCurve = false;

    __host__ static bool valid_parameters(const Parameters& p) {
        return std::isfinite(p.rate);
    }

    __device__ __forceinline__ static Prepared prepare(
        const Parameters& p, const CurveParameters&, float dt
    ) {
        return {p.rate * dt};
    }

    __device__ __forceinline__ static State initial_state(
        const Prepared&
    ) {
        return {0.0f};
    }

    __device__ __forceinline__ static float one_step_transition(
        const Prepared& p, const float*, philox::PhiloxKey,
        std::uint64_t, std::uint32_t, std::uint32_t, State& state
    ) {
        state.integral += p.rate_dt;
        return p.rate_dt;
    }
};

}  // namespace ai_factory::workbench::model::fixed_income::constant_rate
