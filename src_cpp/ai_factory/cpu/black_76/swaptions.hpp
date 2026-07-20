#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::black_76 {

void price_swaption_batch(
    const cuda::Black76SwaptionRow*, std::size_t, cuda::MonteCarloOutput*
);

}  // namespace ai_factory::cpu::black_76
