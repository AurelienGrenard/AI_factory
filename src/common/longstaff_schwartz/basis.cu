// Laguerre basis implementation for the current two-factor LSM regression.
#include "common/longstaff_schwartz/basis.cuh"

#include <cmath>

namespace ai_factory::workbench::longstaff_schwartz {

__device__ __forceinline__ float laguerre_0(float) {
    return 1.0f;
}

__device__ __forceinline__ float laguerre_1(float value) {
    return 1.0f - value;
}

__device__ __forceinline__ float laguerre_2(float value) {
    return fmaf(0.5f * value, value, 1.0f - 2.0f * value);
}

__device__ __forceinline__ TwoFactorLaguerreBasis
TwoFactorLaguerreBasis::evaluate(
    float primary_state,
    float secondary_state
) {
    const float primary_1 = laguerre_1(primary_state);
    return {{
        laguerre_0(primary_state),
        primary_1,
        laguerre_2(primary_state),
        secondary_state,
        secondary_state * secondary_state,
        primary_1 * secondary_state,
    }};
}

}  // namespace ai_factory::workbench::longstaff_schwartz
