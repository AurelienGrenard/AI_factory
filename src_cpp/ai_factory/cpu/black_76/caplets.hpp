#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::black_76 {

void price_caplet_batch(
    const cuda::Black76CapletRow*, std::size_t, cuda::MonteCarloOutput*
);

}  // namespace ai_factory::cpu::black_76
