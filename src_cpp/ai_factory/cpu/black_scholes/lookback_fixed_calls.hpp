#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::black_scholes {

void price_lookback_fixed_call_batch(
    const cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
);

void price_lookback_fixed_call_delta_crn_batch(
    const cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    cuda::PriceDeltaOutput* outputs
);

}  // namespace ai_factory::cpu::black_scholes
