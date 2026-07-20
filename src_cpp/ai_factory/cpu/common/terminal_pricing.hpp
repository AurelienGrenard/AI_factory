#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ai_factory::cpu::terminal_pricing {

template <typename Payoff>
cuda::MonteCarloOutput summarize(
    const std::vector<double>& terminal_spots,
    double strike,
    double discount,
    Payoff payoff
) {
    double sum = 0.0;
    double sumsq = 0.0;
    for (double terminal : terminal_spots) {
        const double value = discount * payoff(terminal, strike);
        sum += value;
        sumsq += value * value;
    }
    const double count = static_cast<double>(terminal_spots.size());
    const double mean = sum / count;
    const double variance = (sumsq - count * mean * mean)
                            / static_cast<double>(terminal_spots.size() - 1U);
    return {mean, std::sqrt(std::max(variance, 0.0) / count)};
}

}  // namespace ai_factory::cpu::terminal_pricing
