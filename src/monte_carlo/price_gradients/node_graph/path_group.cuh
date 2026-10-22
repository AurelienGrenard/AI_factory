// Warp-local path group used by cooperative sensitivity-node evaluators.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail {

template<unsigned int GroupSize>
struct WarpPathGroup {
    static_assert(
        GroupSize == 2U || GroupSize == 4U || GroupSize == 8U
        || GroupSize == 16U || GroupSize == 32U
    );
    static_assert(32U % GroupSize == 0U);

    unsigned int local_lane;
    unsigned int group_in_block;
    unsigned int source_lane;
    unsigned int mask;

    __device__ __forceinline__ static WarpPathGroup make() {
        const unsigned int lane = threadIdx.x & 31U;
        const unsigned int local_lane = lane & (GroupSize - 1U);
        const unsigned int source_lane = lane - local_lane;
        constexpr unsigned int base_mask = GroupSize == 32U
            ? 0xffffffffU
            : (1U << GroupSize) - 1U;
        return {
            local_lane,
            threadIdx.x / GroupSize,
            source_lane,
            base_mask << source_lane,
        };
    }

    __device__ __forceinline__ float broadcast(float value) const {
        return __shfl_sync(mask, value, source_lane);
    }

    template<typename Value>
    __device__ __forceinline__ Value broadcast_value(Value value) const {
        static_assert(std::is_trivially_copyable_v<Value>);
        static_assert(sizeof(Value) % sizeof(std::uint32_t) == 0U);
        union Storage {
            Value value;
            std::uint32_t words[sizeof(Value) / sizeof(std::uint32_t)];
        };
        Storage storage{};
        storage.value = value;
        #pragma unroll
        for (unsigned int word = 0U;
             word < sizeof(Value) / sizeof(std::uint32_t);
             ++word) {
            storage.words[word] = __shfl_sync(
                mask, storage.words[word], source_lane
            );
        }
        return storage.value;
    }

    template<typename Value>
    __device__ __forceinline__ Value broadcast_value(
        Value value,
        unsigned char*
    ) const {
        return broadcast_value(value);
    }

    __device__ __forceinline__ void synchronize() const {
        __syncwarp(mask);
    }
};

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
