#pragma once

#include "ai_factory/cuda/common/autocall.cuh"

#include <cstddef>
#include <stdexcept>

namespace ai_factory::cpu::payoffs {

inline cuda::autocall_detail::PathMetrics autocall_from_path(
    const double* path,
    std::size_t num_steps,
    double spot0,
    double maturity,
    double rate,
    const cuda::AutocallTerms& terms
) {
    if (terms.observation_count == 0U
        || num_steps % terms.observation_count != 0U) {
        throw std::invalid_argument(
            "Autocall observation count must divide the simulation step count."
        );
    }
    const auto stride = num_steps / terms.observation_count;
    cuda::autocall_detail::PathState state{};
    std::size_t call_observation = 0U;
    for (std::size_t observation = 1U;
         observation <= terms.observation_count;
         ++observation) {
        const auto step = observation * stride;
        const double performance = path[step] / spot0;
        const double time = maturity * static_cast<double>(observation)
                            / static_cast<double>(terms.observation_count);
        if (cuda::autocall_detail::observe(
                terms, performance, observation, time, rate, state
            )) {
            call_observation = observation;
            break;
        }
    }
    return cuda::autocall_detail::finish(
        terms,
        path[num_steps] / spot0,
        maturity,
        rate,
        call_observation,
        state
    );
}

inline cuda::autocall_detail::PathMetrics autocall_from_observations(
    const double* observation_spots,
    double spot0,
    double maturity,
    double rate,
    const cuda::AutocallTerms& terms
) {
    if (terms.observation_count == 0U) {
        throw std::invalid_argument(
            "Autocall pricing requires at least one observation."
        );
    }
    cuda::autocall_detail::PathState state{};
    std::size_t call_observation = 0U;
    for (std::size_t observation = 1U;
         observation <= terms.observation_count;
         ++observation) {
        const double time = maturity * static_cast<double>(observation)
                            / static_cast<double>(terms.observation_count);
        if (cuda::autocall_detail::observe(
                terms,
                observation_spots[observation - 1U] / spot0,
                observation,
                time,
                rate,
                state
            )) {
            call_observation = observation;
            break;
        }
    }
    return cuda::autocall_detail::finish(
        terms,
        observation_spots[terms.observation_count - 1U] / spot0,
        maturity,
        rate,
        call_observation,
        state
    );
}

}  // namespace ai_factory::cpu::payoffs
