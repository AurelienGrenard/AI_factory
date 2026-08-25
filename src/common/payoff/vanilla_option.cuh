// Compile-time call and put payoff primitive for any scalar underlying.
#pragma once

#include "common/option_side.cuh"

#include <cmath>

namespace ai_factory::workbench::payoff {

template<OptionSide Side>
__device__ __forceinline__ float vanilla_option_payoff(
    float underlying,
    float strike
) {
    if constexpr (Side == OptionSide::call) {
        return fmaxf(underlying - strike, 0.0f);
    } else {
        return fmaxf(strike - underlying, 0.0f);
    }
}

}  // namespace ai_factory::workbench::payoff
