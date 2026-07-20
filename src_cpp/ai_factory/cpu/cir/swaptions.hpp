#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::cir {
void price_swaption_batch(const cuda::CirSwaptionRow*, std::size_t, std::size_t, double, cuda::MonteCarloOutput*);
}
