#pragma once

#include <algorithm>

namespace ai_factory::cpu::payoffs {

inline double lookback_fixed_call(double running_max, double strike) {
    return std::max(running_max - strike, 0.0);
}

}  // namespace ai_factory::cpu::payoffs
