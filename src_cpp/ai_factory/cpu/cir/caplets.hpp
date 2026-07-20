#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::cir {
void price_caplet_batch(const cuda::CirCapletRow*,std::size_t,std::size_t,double,cuda::MonteCarloOutput*);
}
