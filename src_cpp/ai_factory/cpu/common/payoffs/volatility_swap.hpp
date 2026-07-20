#pragma once

namespace ai_factory::cpu::payoffs {

inline double volatility_swap(double realized_volatility, double strike) {
    return realized_volatility - strike;
}

}  // namespace ai_factory::cpu::payoffs
