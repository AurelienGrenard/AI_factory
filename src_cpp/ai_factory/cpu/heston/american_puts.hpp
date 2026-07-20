#pragma once

#include "ai_factory/cuda/common/types.cuh"
#include "ai_factory/cpu/heston/common.hpp"

#include <cstddef>
#include <vector>

namespace ai_factory::cpu::heston {

void price_american_put_from_paths(
    const simulation::HestonStatePaths& paths,
    std::size_t num_paths,
    std::size_t num_steps,
    double strike,
    double theta,
    double maturity,
    double rate,
    ai_factory::cuda::MonteCarloOutput& output
);

void price_american_put(
    const cuda::HestonRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput& output
);

void price_american_put_batch(
    const cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
);

}  // namespace ai_factory::cpu::heston
