// Model-independent shell for one-factor affine zero-coupon bonds.
#pragma once

#include <cuda_runtime.h>

namespace ai_factory::workbench::fixed_income {

struct OneFactorAffineBondCoefficients {
    float log_A;
    float B;
};

// Return log P(t,T) = log A(t,T) - B(t,T) x_t.
template<typename Provider, typename Parameters>
__device__ __forceinline__ float log_zero_coupon_bond(
    const Provider& provider,
    const Parameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    const OneFactorAffineBondCoefficients coefficients =
        provider.affine_bond_coefficients(
            parameters, valuation_time, maturity
        );
    return fmaf(
        -coefficients.B,
        state,
        coefficients.log_A
    );
}

// Return P(t,T) from the shared log-affine representation.
template<typename Provider, typename Parameters>
__device__ __forceinline__ float zero_coupon_bond(
    const Provider& provider,
    const Parameters& parameters,
    float state,
    float valuation_time,
    float maturity
) {
    return expf(log_zero_coupon_bond(
        provider, parameters, state, valuation_time, maturity
    ));
}

}  // namespace ai_factory::workbench::fixed_income
