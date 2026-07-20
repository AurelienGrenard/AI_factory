#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::cir_plus_plus {
void price_swaption_batch(const cuda::CirPlusPlusSwaptionRow*, std::size_t, std::size_t, double, cuda::MonteCarloOutput*);
}
