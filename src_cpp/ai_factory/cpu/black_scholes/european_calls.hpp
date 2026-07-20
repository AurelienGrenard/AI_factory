#pragma once
#include "ai_factory/cuda/common/types.cuh"
namespace ai_factory::cpu::black_scholes {
void price_european_call_batch(const cuda::BlackScholesRow*, std::size_t, std::size_t, std::size_t, cuda::MonteCarloOutput*);
}
