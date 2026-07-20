#pragma once
#include "ai_factory/cuda/common/barrier_pricing.cuh"
#include "ai_factory/cuda/heston/dynamics.cuh"
namespace ai_factory::cuda::heston_detail {
template <bool Up>
struct BarrierSimulator {
    __device__ static barrier_detail::PathState run(const HestonRow& row, std::size_t path, std::size_t steps, double barrier) {
        const auto state = simulate_heston_qe_barrier<Up>(row, steps, path, barrier);
        return {state.terminal_spot, state.hit};
    }
};
}
