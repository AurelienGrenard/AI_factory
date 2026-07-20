#pragma once
#include "ai_factory/cuda/common/types.cuh"
namespace ai_factory::cpu::rough_bergomi {
void price_european_call_batch(const cuda::RoughBergomiRow*, std::size_t, std::size_t, std::size_t, cuda::MonteCarloOutput*);
}
