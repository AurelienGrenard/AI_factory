#pragma once

#include <algorithm>

namespace ai_factory::cpu::payoffs {

inline double asian_arithmetic_call(double average_spot, double strike) {
    return std::max(average_spot - strike, 0.0);
}

}  // namespace ai_factory::cpu::payoffs
