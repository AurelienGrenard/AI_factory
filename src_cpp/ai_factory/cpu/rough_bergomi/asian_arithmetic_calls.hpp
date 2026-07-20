#pragma once

#include "ai_factory/cuda/common/types.cuh"

#include <cstddef>

namespace ai_factory::cpu::rough_bergomi {

void price_asian_arithmetic_call(
    const cuda::RoughBergomiRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput& output
);

void price_asian_arithmetic_call_delta_crn(
    const cuda::RoughBergomiRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    cuda::PriceDeltaOutput& output
);

void price_asian_arithmetic_call_batch(
    const cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
);

void price_asian_arithmetic_call_delta_crn_batch(
    const cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    cuda::PriceDeltaOutput* outputs
);

}  // namespace ai_factory::cpu::rough_bergomi
