// Reusable host-side construction of Vasicek model datasets.
#pragma once

#include "tools/datasets/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::datasets::vasicek {

// Hold one closed interval used by the Vasicek sampler.
struct SamplingRange {
    float minimum;
    float maximum;
};

// Define the Vasicek process before reconstructing instantaneous volatility.
struct ProcessGenerationBounds {
    SamplingRange mean_reversion;
    SamplingRange long_term_mean;
    SamplingRange stationary_standard_deviation;
};

// Add the initial state required by a standalone Vasicek model.
struct GenerationBounds {
    ProcessGenerationBounds process;
    SamplingRange initial_state;
};

// Sample reusable Vasicek process parameters without one initial state.
GeneratedRows generate_process_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const ProcessGenerationBounds& bounds
);

// Sample stable Vasicek rows and reconstruct sigma from stationary dispersion.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
);

}  // namespace ai_factory::workbench::datasets::vasicek
