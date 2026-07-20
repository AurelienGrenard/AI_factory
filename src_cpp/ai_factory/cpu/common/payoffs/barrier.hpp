#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ai_factory::cpu::payoffs {

template <bool Up, bool KnockIn>
cuda::MonteCarloOutput barrier_call_from_paths(
    const std::vector<double>& paths,
    std::size_t num_paths,
    std::size_t num_steps,
    double strike,
    double barrier,
    double discount
) {
    double sum = 0.0;
    double sumsq = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum, sumsq) schedule(static)
#endif
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        const auto offset = path * (num_steps + 1U);
        bool hit = false;
        for (std::size_t step = 1U; step <= num_steps; ++step) {
            const double spot = paths[offset + step];
            hit = hit || (Up ? spot >= barrier : spot <= barrier);
        }
        const bool active = KnockIn ? hit : !hit;
        const double payoff = active
            ? discount * std::max(paths[offset + num_steps] - strike, 0.0)
            : 0.0;
        sum += payoff;
        sumsq += payoff * payoff;
    }
    const double count = static_cast<double>(num_paths);
    const double mean = sum / count;
    const double variance =
        (sumsq - count * mean * mean) / static_cast<double>(num_paths - 1U);
    return {mean, std::sqrt(std::max(variance, 0.0)) / std::sqrt(count)};
}

}  // namespace ai_factory::cpu::payoffs
