#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::black_scholes {

void price_volatility_swap_batch(
    const cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
);

}  // namespace ai_factory::cpu::black_scholes
