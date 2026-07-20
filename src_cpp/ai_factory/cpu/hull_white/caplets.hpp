#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::hull_white {
void price_caplet_batch(const cuda::HullWhiteCapletRow*,std::size_t,std::size_t,cuda::MonteCarloOutput*);
}
