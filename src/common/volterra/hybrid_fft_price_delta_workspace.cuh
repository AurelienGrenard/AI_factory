// Paired FFT pricing adds only delta partial moments to the original workspace.
#pragma once
#include "common/volterra/hybrid_fft_workspace.cuh"

namespace ai_factory::workbench::volterra {

inline std::size_t required_hybrid_fft_price_delta_workspace_bytes(
    std::size_t steps, std::size_t paths, std::size_t chunk) {
    return checked_hybrid_fft_sum(required_hybrid_fft_workspace_bytes(steps, paths, chunk),
        checked_hybrid_fft_product(hybrid_fft_partial_moment_count(paths), 2U * sizeof(double),
                                  "FFT delta partial bytes overflow."),
        "FFT price-delta workspace bytes overflow.");
}

inline HybridFftWorkspacePlan plan_hybrid_fft_price_delta_workspace(
    std::size_t steps, std::size_t paths, std::size_t chunk) {
    auto plan = plan_hybrid_fft_workspace(steps, paths, chunk);
    plan.workspace_bytes = required_hybrid_fft_price_delta_workspace_bytes(steps, paths, chunk);
    return plan;
}

}  // namespace ai_factory::workbench::volterra
