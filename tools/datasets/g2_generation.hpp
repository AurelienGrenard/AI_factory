// Reusable host-side construction of G2 and G2++ model datasets.
#pragma once

#include "tools/datasets/sampling.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::datasets::g2 {

// Hold one closed interval used by the G2 sampler.
struct SamplingRange {
    float minimum;
    float maximum;
};

// Define stable two-factor dynamics before reconstructing both volatilities.
struct ProcessGenerationBounds {
    SamplingRange mean_reversion_x;
    SamplingRange mean_reversion_gap;
    SamplingRange stationary_standard_deviation_x;
    SamplingRange stationary_standard_deviation_y;
    SamplingRange correlation;
};

// Add both initial factor states required by standalone G2.
struct GenerationBounds {
    ProcessGenerationBounds process;
    SamplingRange initial_state_x;
    SamplingRange initial_state_y;
};

// Sample reusable G2 dynamics without choosing initial factor states.
GeneratedRows generate_process_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const ProcessGenerationBounds& bounds
);

// Sample stable G2 rows and reconstruct both instantaneous volatilities.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
);

}  // namespace ai_factory::workbench::datasets::g2
