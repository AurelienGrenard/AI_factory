// Warp- or block-local team assigned to one Monte Carlo path.
#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail {

namespace cg = cooperative_groups;

template<unsigned int TeamSize>
struct PathTeam {
    static_assert(TeamSize >= 2U && TeamSize <= 1024U);
    static_assert((TeamSize & (TeamSize - 1U)) == 0U);

    unsigned int local_lane;
    unsigned int team_in_block;
    unsigned int warp_mask;

    __device__ __forceinline__ static PathTeam make() {
        const auto local_lane = threadIdx.x & (TeamSize - 1U);
        unsigned int mask = 0xffffffffU;
        if constexpr (TeamSize < 32U) {
            const auto warp_lane = threadIdx.x & 31U;
            const auto first_lane = warp_lane - local_lane;
            mask = ((1U << TeamSize) - 1U) << first_lane;
        }
        return {
            local_lane,
            threadIdx.x / TeamSize,
            mask,
        };
    }

    __device__ __forceinline__ void synchronize() const {
        if constexpr (TeamSize <= 32U) {
            __syncwarp(warp_mask);
        } else {
            auto tile = cg::tiled_partition<TeamSize>(
                cg::this_thread_block()
            );
            tile.sync();
        }
    }

    template<typename Value>
    __device__ __forceinline__ Value broadcast_value(
        Value value,
        unsigned char* team_scratch
    ) const {
        static_assert(std::is_trivially_copyable_v<Value>);
        if constexpr (TeamSize <= 32U) {
            static_assert(sizeof(Value) % sizeof(std::uint32_t) == 0U);
            union Storage {
                Value value;
                std::uint32_t words[sizeof(Value) / sizeof(std::uint32_t)];
            } storage{};
            storage.value = value;
            const auto warp_lane = threadIdx.x & 31U;
            const auto source_lane = warp_lane - local_lane;
            #pragma unroll
            for (unsigned int word = 0U;
                 word < sizeof(Value) / sizeof(std::uint32_t);
                 ++word) {
                storage.words[word] = __shfl_sync(
                    warp_mask, storage.words[word], source_lane
                );
            }
            return storage.value;
        } else {
            static_assert(alignof(Value) <= 16U);
            auto* shared_value = reinterpret_cast<Value*>(team_scratch);
            if (local_lane == 0U) *shared_value = value;
            synchronize();
            value = *shared_value;
            synchronize();
            return value;
        }
    }
};

template<std::size_t Size>
inline constexpr std::size_t align_path_team_scratch_v =
    (Size + 15U) & ~std::size_t{15U};

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
