// Reusable host-side construction of Nelson-Siegel parameter datasets.
#pragma once

#include "tools/datasets/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::curve {

// Hold one closed interval used by the Nelson-Siegel sampler.
struct NelsonSiegelSamplingRange {
    float minimum;
    float maximum;
};

// Define the sampled forward levels and the accepted global forward range.
struct NelsonSiegelGenerationBounds {
    NelsonSiegelSamplingRange long_forward;
    NelsonSiegelSamplingRange short_forward;
    NelsonSiegelSamplingRange medium_forward;
    NelsonSiegelSamplingRange tau;
    NelsonSiegelSamplingRange accepted_forward;
};

// Reconstruct accepted Nelson-Siegel rows from interpretable forward levels.
datasets::GeneratedRows generate_nelson_siegel_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const NelsonSiegelGenerationBounds& bounds
);

}  // namespace ai_factory::workbench::curve
