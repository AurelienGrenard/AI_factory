// Reusable host-side construction of constrained Svensson curve datasets.
#pragma once

#include "tools/datasets/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::datasets::svensson {

// Hold one closed interval used by the Svensson sampler.
struct SamplingRange {
    float minimum;
    float maximum;
};

// Define sampled forward levels, decay scales, and accepted curve bounds.
struct GenerationBounds {
    SamplingRange long_forward;
    SamplingRange short_forward;
    SamplingRange first_medium_forward;
    SamplingRange second_medium_forward;
    SamplingRange tau1;
    SamplingRange tau2;
    SamplingRange accepted_forward;
    SamplingRange accepted_curvature;
};

// Reconstruct accepted Svensson rows from four interpretable forward levels.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
);

}  // namespace ai_factory::workbench::datasets::svensson
