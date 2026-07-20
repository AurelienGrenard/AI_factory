#pragma once
#include "ai_factory/cuda/common/types.cuh"
namespace ai_factory::cpu::rough_heston {
void price_european_call_batch(const cuda::RoughHestonRow*, std::size_t, std::size_t, std::size_t, cuda::MonteCarloOutput*);
}
