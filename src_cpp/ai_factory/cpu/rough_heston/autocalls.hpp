#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::rough_heston {

void price_autocall_batch(
    const cuda::RoughHestonAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::AutocallOutput* outputs
);

}  // namespace ai_factory::cpu::rough_heston
