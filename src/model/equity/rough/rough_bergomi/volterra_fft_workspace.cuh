// Public workspace planning shared by all rough-Bergomi FFT products.
#pragma once

#include "common/volterra/hybrid_fft_workspace.cuh"

#include <cstddef>

namespace ai_factory::workbench::model::equity::rough_bergomi {

using WorkspacePlan = volterra::HybridFftWorkspacePlan;

inline WorkspacePlan plan_pricing_workspace(
    std::size_t maximum_step_count,
    std::size_t monte_carlo_paths_per_price,
    std::size_t path_chunk_size
) {
    return volterra::plan_hybrid_fft_workspace(
        maximum_step_count,
        monte_carlo_paths_per_price,
        path_chunk_size
    );
}

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
