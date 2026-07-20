#pragma once

#include "ai_factory/cuda/black_scholes/dynamics.cuh"
#include "ai_factory/cuda/common/barrier_pricing.cuh"

namespace ai_factory::cuda::black_scholes_detail {

template <bool Up>
struct BarrierSimulator {
    __device__ static barrier_detail::PathState run(
        const BlackScholesRow& row,
        std::size_t path,
        std::size_t num_steps,
        double barrier
    ) {
        const double dt = row.maturity / static_cast<double>(num_steps);
        const double sqrt_dt = sqrt(dt);
        rng::NormalSequence normals(row.seed, 0U, path * num_steps);
        double spot = row.spot;
        bool hit = false;
        for (std::size_t step = 0; step < num_steps; ++step) {
            spot *= exp(log_step(row, dt, sqrt_dt, normals.next()));
            hit = hit || (Up ? spot >= barrier : spot <= barrier);
        }
        return {spot, hit};
    }
};

}  // namespace ai_factory::cuda::black_scholes_detail
