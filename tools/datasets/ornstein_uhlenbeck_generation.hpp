// Reusable host-side construction of Ornstein-Uhlenbeck model datasets.
#pragma once

#include "tools/datasets/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::datasets::ornstein_uhlenbeck {

// Hold one closed interval used by the OU sampler.
struct SamplingRange {
    float minimum;
    float maximum;
};

// Define the OU dynamics before reconstructing instantaneous volatility.
struct DynamicsGenerationBounds {
    SamplingRange mean_reversion;
    SamplingRange stationary_standard_deviation;
};

// Add the initial state required by a standalone OU model.
struct GenerationBounds {
    DynamicsGenerationBounds dynamics;
    SamplingRange initial_state;
};

// Sample reusable OU dynamics without imposing one initial state.
GeneratedRows generate_dynamics_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const DynamicsGenerationBounds& bounds
);

// Sample stable OU rows and reconstruct sigma from stationary dispersion.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
);

}  // namespace ai_factory::workbench::datasets::ornstein_uhlenbeck
