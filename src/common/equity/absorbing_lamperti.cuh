// Absorbing CEV evolution in the Lamperti coordinate used by SABR models.
#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>

namespace ai_factory::workbench::equity {

// Advance log(S) through Y = S^(1-beta)/(1-beta). A non-positive finite
// proposal hits the attainable absorbing boundary. Non-finite arithmetic is
// deliberately propagated instead of being hidden behind a positive floor.
__device__ __forceinline__ void advance_absorbing_lamperti_log_spot(
    float beta,
    float one_minus_beta,
    float drift_time_step,
    float time_step,
    float sqrt_time_step,
    float volatility,
    float normal,
    float& log_spot
) {
    if (log_spot == -CUDART_INF_F) return;

    const float spot = expf(log_spot);
    if (spot == 0.0f) {
        log_spot = -CUDART_INF_F;
        return;
    }

    // Work with the relative Lamperti increment. Reconstructing
    // log((1-beta) * proposal)/(1-beta) subtracts nearly equal FP32 numbers
    // when beta is close to one and creates a persistent positive spot bias.
    const float power = expf(one_minus_beta * log_spot);
    const float relative_increment = one_minus_beta * (
        drift_time_step
        + volatility * sqrt_time_step * normal / power
        - 0.5f * beta * volatility * volatility * time_step
            / (power * power));
    if (relative_increment > -1.0f) {
        log_spot += log1pf(relative_increment) / one_minus_beta;
    } else if (relative_increment <= -1.0f) {
        log_spot = -CUDART_INF_F;
    } else {
        log_spot = relative_increment;
    }
}

}  // namespace ai_factory::workbench::equity
