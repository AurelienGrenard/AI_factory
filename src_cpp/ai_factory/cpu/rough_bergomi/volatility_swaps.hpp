#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::rough_bergomi {

void price_volatility_swap(
    const cuda::RoughBergomiRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput& output
);

void price_volatility_swap_batch(
    const cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
);

}  // namespace ai_factory::cpu::rough_bergomi
