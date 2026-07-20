#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::heston {

void price_autocall_batch(
    const cuda::HestonAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::AutocallOutput* outputs
);

}  // namespace ai_factory::cpu::heston
