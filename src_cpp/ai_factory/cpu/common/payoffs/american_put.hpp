#pragma once

#include <algorithm>

namespace ai_factory::cpu::payoffs {

inline double american_put_immediate(double spot, double strike) {
    return std::max(strike - spot, 0.0);
}

}  // namespace ai_factory::cpu::payoffs
