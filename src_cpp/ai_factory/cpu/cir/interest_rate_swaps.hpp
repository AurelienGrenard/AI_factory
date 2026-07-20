#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::cir {
void price_interest_rate_swap_batch(const cuda::CirSwapRow*, std::size_t, cuda::MonteCarloOutput*);
}
