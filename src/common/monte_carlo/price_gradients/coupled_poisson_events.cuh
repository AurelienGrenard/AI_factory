// Nested compound-Poisson events shared by jump-model sensitivity adapters.
#pragma once

#include "common/compound_poisson.cuh"
#include "common/philox_domains.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::monte_carlo::price_gradients {

struct StandardNormalEventMarks {
    using Mark = float;
    using Cache = philox::NormalPairCache;

    __device__ __forceinline__ static Mark draw(
        philox::DomainUniformSequence& uniforms,
        Cache& cache
    ) {
        return philox::next_normal(uniforms, cache);
    }
};

struct UniformPairEventMarks {
    struct Mark {
        float first;
        float second;
    };
    struct Cache {};

    __device__ __forceinline__ static Mark draw(
        philox::DomainUniformSequence& uniforms,
        Cache&
    ) {
        return {uniforms.next(), uniforms.next()};
    }
};

// Node zero is the central scenario. Its events are drawn first and replayed
// into every node with at least the central integrated intensity. Lower nodes
// retain central events through one shared thinning uniform. Higher nodes read
// one common exponential-arrival process beyond the central intensity. The callback
// transforms one primitive mark for one accepted node; it therefore remains
// model-specific while the count coupling and nesting stay common.
template<
    std::size_t NodeCapacity,
    typename EventMarks,
    typename ApplyMark>
__device__ __forceinline__ void replay_coupled_poisson_events(
    philox::DomainRandomContext& random,
    std::uint32_t step,
    const float (&means)[NodeCapacity],
    std::uint8_t node_count,
    std::uint32_t (&counts)[NodeCapacity],
    ApplyMark apply_mark
) {
    #pragma unroll
    for (unsigned int node = 0U; node < NodeCapacity; ++node) {
        counts[node] = 0U;
    }

    const float central_mean = means[0U];
    auto count_uniforms = random.source<
        compound_poisson::kCountSource
    >(step);
    const std::uint32_t central_count = compound_poisson::draw_count(
        count_uniforms, central_mean, expf(-central_mean)
    );

    bool has_lower_mean = false;
    for (std::uint8_t node = 1U; node < node_count; ++node) {
        has_lower_mean = has_lower_mean || means[node] < central_mean;
    }
    auto central_marks = random.source<
        compound_poisson::kMarkSource
    >(step);
    typename EventMarks::Cache central_mark_cache{};
    auto thinning_uniforms = random.source<
        compound_poisson::kThinningSource
    >(step);
    for (std::uint32_t event = 0U; event < central_count; ++event) {
        const auto mark = EventMarks::draw(
            central_marks, central_mark_cache
        );
        const float thinning = has_lower_mean
            ? thinning_uniforms.next()
            : 0.0f;
        for (std::uint8_t node = 0U; node < node_count; ++node) {
            const bool retained = node == 0U || means[node] >= central_mean
                || (central_mean > 0.0f
                    && thinning < means[node] / central_mean);
            if (retained) {
                ++counts[node];
                apply_mark(node, mark);
            }
        }
    }

    float maximum_mean = central_mean;
    for (std::uint8_t node = 1U; node < node_count; ++node) {
        if (means[node] > maximum_mean) maximum_mean = means[node];
    }
    auto extension_arrivals = random.source<
        compound_poisson::kExtensionArrivalSource
    >(step);
    auto extension_marks = random.source<
        compound_poisson::kExtensionMarkSource
    >(step);
    typename EventMarks::Cache extension_mark_cache{};
    float arrival = central_mean;
    while (arrival < maximum_mean) {
        arrival -= logf(extension_arrivals.next());
        if (arrival > maximum_mean) break;
        const auto mark = EventMarks::draw(
            extension_marks, extension_mark_cache
        );
        for (std::uint8_t node = 1U; node < node_count; ++node) {
            if (means[node] >= arrival) {
                ++counts[node];
                apply_mark(node, mark);
            }
        }
    }
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
