#pragma once

#include "ai_factory/cuda/common/types.cuh"
#include "ai_factory/cpu/rough_heston/common.hpp"

#include <cstddef>
#include <vector>

namespace ai_factory::cpu::rough_heston {

void price_american_put_from_paths(
    const simulation::RoughHestonStatePaths& paths,
    std::size_t num_paths,
    std::size_t num_steps,
    double strike,
    double theta,
    double maturity,
    double rate,
    ai_factory::cuda::MonteCarloOutput& output
);

void price_american_put(
    const cuda::RoughHestonRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput& output
);

void price_american_put_batch(
    const cuda::RoughHestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
);

}  // namespace ai_factory::cpu::rough_heston
