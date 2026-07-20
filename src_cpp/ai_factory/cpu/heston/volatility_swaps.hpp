#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::heston {

void price_volatility_swap(
    const cuda::HestonRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput& output
);

void price_volatility_swap_batch(
    const cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
);

}  // namespace ai_factory::cpu::heston
