#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::g2_plus_plus {
void price_interest_rate_swap_batch(const cuda::G2PlusPlusSwapRow*,std::size_t,cuda::MonteCarloOutput*);
}
