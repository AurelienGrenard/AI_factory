// Reusable host-side construction of Nelson-Siegel parameter datasets.
#pragma once

#include "tools/datasets/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::datasets::nelson_siegel {

// Hold one closed interval used by the Nelson-Siegel sampler.
struct SamplingRange {
    float minimum;
    float maximum;
};

// Define the sampled forward levels and the accepted global forward range.
struct GenerationBounds {
    SamplingRange long_forward;
    SamplingRange short_forward;
    SamplingRange medium_forward;
    SamplingRange tau;
    SamplingRange accepted_forward;
};

// Reconstruct accepted Nelson-Siegel rows from interpretable forward levels.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
);

}  // namespace ai_factory::workbench::datasets::nelson_siegel
