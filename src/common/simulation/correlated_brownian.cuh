// Fixed-consumption correlated Gaussian shocks shared by multi-asset paths.
#pragma once

#include "common/philox_domains.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::simulation {

struct NormalQuad {
    float value[4];
};

// Random access uses exactly the counter and Box-Muller pairing of
// BrownianNormals::generate. It is needed by FFT tiles that visit time cells
// out of the order used by the joint path kernel.
__device__ __forceinline__ float independent_brownian_normal(
    philox::PhiloxKey key, std::uint64_t path,
    std::uint32_t step, std::size_t factor
) {
    const auto domain = philox::source_step_domain<1U>(step);
    const auto uniforms = philox::domain_uniform_quad(
        key, path, static_cast<std::uint32_t>(factor / 4U), domain
    );
    const auto pair = (factor & 2U) == 0U
        ? philox::box_muller(uniforms.first, uniforms.second)
        : philox::box_muller(uniforms.third, uniforms.fourth);
    return (factor & 1U) == 0U ? pair.first : pair.second;
}

__device__ __forceinline__ float correlated_brownian_factor(
    philox::PhiloxKey key, std::uint64_t path,
    std::uint32_t step, std::size_t factor,
    const float* packed_lower
) {
    const std::size_t offset = factor * (factor + 1U) / 2U;
    float result = 0.0f;
    for (std::size_t column = 0U; column <= factor; ++column) {
        result = fmaf(
            packed_lower[offset + column],
            independent_brownian_normal(key, path, step, column), result
        );
    }
    return result;
}

// Internal model correlations are recovered once per price from the same
// Cholesky rows that drive its shocks.
__device__ __forceinline__ float factor_correlation(
    const float* packed_lower, std::size_t first, std::size_t second
) {
    const std::size_t first_offset = first * (first + 1U) / 2U;
    const std::size_t second_offset = second * (second + 1U) / 2U;
    const std::size_t common = first < second ? first : second;
    double value = 0.0;
    for (std::size_t column = 0U; column <= common; ++column) {
        value += static_cast<double>(packed_lower[first_offset + column])
            * static_cast<double>(packed_lower[second_offset + column]);
    }
    return static_cast<float>(value);
}

template<unsigned int TeamSize>
__device__ __forceinline__ unsigned int subgroup_mask() {
    static_assert(TeamSize > 0U && TeamSize <= 32U
        && (TeamSize & (TeamSize - 1U)) == 0U);
    constexpr unsigned int bits = 0xFFFFFFFFU >> (32U - TeamSize);
    const unsigned int first = (threadIdx.x & 31U) & ~(TeamSize - 1U);
    return bits << first;
}

// The Brownian domain is fixed-consumption. One group produces four normals,
// and group numbers are independent of the path-team size.
template<std::size_t Dimension, unsigned int TeamSize>
struct BrownianNormals {
    static_assert(Dimension > 0U);
    static_assert(TeamSize > 0U && TeamSize <= 32U
        && (TeamSize & (TeamSize - 1U)) == 0U);
    static constexpr std::size_t kGroups = (Dimension + 3U) / 4U;
    static constexpr std::size_t kLocalGroups =
        (kGroups + TeamSize - 1U) / TeamSize;

    NormalQuad local[kLocalGroups];

    __device__ __forceinline__ void generate(
        philox::PhiloxKey key,
        std::uint64_t path,
        std::uint32_t step,
        unsigned int lane
    ) {
        const auto domain = philox::source_step_domain<1U>(step);
        for (std::size_t group = lane; group < kGroups; group += TeamSize) {
            const auto uniforms = philox::domain_uniform_quad(
                key, path, static_cast<std::uint32_t>(group), domain
            );
            const auto first = philox::box_muller(
                uniforms.first, uniforms.second
            );
            const auto second = philox::box_muller(
                uniforms.third, uniforms.fourth
            );
            local[group / TeamSize] = {{
                first.first, first.second, second.first, second.second
            }};
        }
    }

    __device__ __forceinline__ void store_to_shared(
        float* shared_normals,unsigned int lane
    ) const {
        for(std::size_t group=lane;group<kGroups;group+=TeamSize) {
            #pragma unroll
            for(unsigned position=0U;position<4U;++position) {
                const auto factor=group*4U+position;
                if(factor<Dimension)
                    shared_normals[factor]=
                        local[group/TeamSize].value[position];
            }
        }
        if constexpr (TeamSize>1U)
            __syncwarp(subgroup_mask<TeamSize>());
    }

    __device__ __forceinline__ float get(
        std::size_t factor, unsigned int lane
    ) const {
        const std::size_t group = factor / 4U;
        const unsigned int owner = static_cast<unsigned int>(group % TeamSize);
        const float value = lane == owner
            ? local[group / TeamSize].value[factor % 4U] : 0.0f;
        if constexpr (TeamSize == 1U) {
            return value;
        } else {
            return __shfl_sync(subgroup_mask<TeamSize>(), value, owner, TeamSize);
        }
    }

};

template<std::size_t Dimension, unsigned int TeamSize>
struct CorrelatedBrownianStep {
    float local[(Dimension + TeamSize - 1U) / TeamSize];
    const float* shared_factors=nullptr;

    __device__ __forceinline__ bool has_shared_factors() const {
        return shared_factors!=nullptr;
    }

    // The input normal vector is no longer read after the triangular
    // product, so its team buffer can hold the correlated factors.
    __device__ __forceinline__ void publish_to_shared(
        float* shared_vector,unsigned lane
    ) {
        if constexpr (TeamSize>1U)
            __syncwarp(subgroup_mask<TeamSize>());
        for(std::size_t row=lane;row<Dimension;row+=TeamSize)
            shared_vector[row]=local[row/TeamSize];
        if constexpr (TeamSize>1U)
            __syncwarp(subgroup_mask<TeamSize>());
        shared_factors=shared_vector;
    }

    __device__ __forceinline__ void calculate(
        const BrownianNormals<Dimension, TeamSize>& normals,
        const float* packed_lower,
        unsigned int lane
    ) {
        // A lane owns every TeamSize-th output row. Lanes broadcast the
        // same input column together, while their row products run in
        // parallel. The bound is the last row of the current lane group.
        constexpr std::size_t local_rows =
            (Dimension + TeamSize - 1U) / TeamSize;
        for (std::size_t local_row = 0; local_row < local_rows;
             ++local_row) {
            const std::size_t row = local_row * TeamSize + lane;
            const std::size_t last =
                (local_row + 1U) * TeamSize < Dimension
                    ? (local_row + 1U) * TeamSize - 1U
                    : Dimension - 1U;
            const std::size_t offset = row * (row + 1U) / 2U;
            float value = 0.0f;
            for (std::size_t column = 0; column <= last; ++column) {
                const float normal = normals.get(column, lane);
                if (row < Dimension && column <= row) {
                    value = fmaf(
                        packed_lower[offset + column], normal, value
                    );
                }
            }
            if (row < Dimension) local[local_row] = value;
        }
    }

    __device__ __forceinline__ void calculate_from_shared(
        const float* normals,const float* packed_lower,
        unsigned int lane
    ) {
        constexpr std::size_t local_rows=
            (Dimension+TeamSize-1U)/TeamSize;
        for(std::size_t local_row=0U;local_row<local_rows;++local_row) {
            const std::size_t row=local_row*TeamSize+lane;
            const std::size_t last=
                (local_row+1U)*TeamSize<Dimension
                    ? (local_row+1U)*TeamSize-1U
                    : Dimension-1U;
            const std::size_t offset=row*(row+1U)/2U;
            float value=0.0f;
            for(std::size_t column=0U;column<=last;++column) {
                const float normal=normals[column];
                if(row<Dimension && column<=row)
                    value=fmaf(packed_lower[offset+column],
                               normal,value);
            }
            if(row<Dimension) local[local_row]=value;
        }
    }

    __device__ __forceinline__ float get(
        std::size_t factor, unsigned int lane
    ) const {
        if(shared_factors) return shared_factors[factor];
        const unsigned int owner =
            static_cast<unsigned int>(factor % TeamSize);
        const float value = lane == owner
            ? local[factor / TeamSize] : 0.0f;
        if constexpr (TeamSize == 1U) {
            return value;
        } else {
            return __shfl_sync(subgroup_mask<TeamSize>(), value, owner, TeamSize);
        }
    }
};

}  // namespace ai_factory::workbench::simulation
