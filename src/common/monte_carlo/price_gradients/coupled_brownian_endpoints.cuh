// Nested Brownian endpoints for direct-transition maturity sensitivities.
#pragma once

#include "common/philox_domains.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

// Node zero is the central horizon. Shorter endpoints are Brownian bridges
// constructed from it in descending time order; longer endpoints append
// independent increments in ascending time order. Equal horizons reuse the
// exact central normal, preserving central and fixed-horizon bit patterns.
template<std::size_t NodeCapacity, typename Horizon, typename Store>
__device__ __forceinline__ void draw_coupled_brownian_normals(
    philox::DomainUniformSequence& uniforms,
    philox::NormalPairCache& cache,
    std::uint16_t node_count,
    Horizon horizon,
    Store store
) {
    float endpoints[NodeCapacity]{};
    const float central_time_years = horizon(0U);
    const float central_normal = philox::next_normal(uniforms, cache);
    endpoints[0U] = sqrtf(central_time_years) * central_normal;

    std::uint16_t representatives[NodeCapacity]{};
    std::uint16_t lower[NodeCapacity]{};
    std::uint16_t upper[NodeCapacity]{};
    std::uint16_t lower_count = 0U;
    std::uint16_t upper_count = 0U;
    for (std::uint16_t node = 1U; node < node_count; ++node) {
        const float time_years = horizon(node);
        representatives[node] = node;
        for (std::uint16_t known = 0U; known < node; ++known) {
            if (horizon(known) == time_years) {
                representatives[node] = known;
                break;
            }
        }
        if (representatives[node] != node) continue;
        if (time_years < central_time_years) lower[lower_count++] = node;
        else if (time_years > central_time_years) upper[upper_count++] = node;
        else endpoints[node] = endpoints[0U];
    }

    for (std::uint16_t index = 1U; index < lower_count; ++index) {
        const auto node = lower[index];
        std::uint16_t position = index;
        while (position > 0U
               && horizon(lower[position - 1U]) < horizon(node)) {
            lower[position] = lower[position - 1U];
            --position;
        }
        lower[position] = node;
    }
    float right_time_years = central_time_years;
    float right_endpoint = endpoints[0U];
    for (std::uint16_t index = 0U; index < lower_count; ++index) {
        const auto node = lower[index];
        const float time_years = horizon(node);
        const float fraction = time_years / right_time_years;
        endpoints[node] = fraction * right_endpoint
            + sqrtf(
                time_years * (right_time_years - time_years)
                / right_time_years
            )
                * philox::next_normal(uniforms, cache);
        right_time_years = time_years;
        right_endpoint = endpoints[node];
    }

    for (std::uint16_t index = 1U; index < upper_count; ++index) {
        const auto node = upper[index];
        std::uint16_t position = index;
        while (position > 0U
               && horizon(upper[position - 1U]) > horizon(node)) {
            upper[position] = upper[position - 1U];
            --position;
        }
        upper[position] = node;
    }
    float left_time_years = central_time_years;
    float left_endpoint = endpoints[0U];
    for (std::uint16_t index = 0U; index < upper_count; ++index) {
        const auto node = upper[index];
        const float time_years = horizon(node);
        endpoints[node] = left_endpoint
            + sqrtf(time_years - left_time_years)
                * philox::next_normal(uniforms, cache);
        left_time_years = time_years;
        left_endpoint = endpoints[node];
    }

    for (std::uint16_t node = 1U; node < node_count; ++node) {
        if (representatives[node] != node) {
            endpoints[node] = endpoints[representatives[node]];
        }
    }

    for (std::uint16_t node = 0U; node < node_count; ++node) {
        store(
            node,
            horizon(node) == central_time_years
                ? central_normal
                : endpoints[node] / sqrtf(horizon(node))
        );
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
