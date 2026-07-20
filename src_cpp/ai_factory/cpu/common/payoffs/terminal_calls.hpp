#pragma once

#include <algorithm>

namespace ai_factory::payoffs {

inline double european_call(double terminal_spot, double strike) {
    return std::max(terminal_spot - strike, 0.0);
}

inline double digital_call(double terminal_spot, double strike) {
    return terminal_spot > strike ? 1.0 : 0.0;
}

}  // namespace ai_factory::payoffs
