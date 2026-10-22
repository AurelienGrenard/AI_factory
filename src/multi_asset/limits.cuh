// Supported Brownian dimension for scans that keep packed Cholesky in shared.
#pragma once

#include <cstddef>

namespace ai_factory::workbench::monte_carlo {

// At 128 factors, packed FP32 L uses 33,024 bytes. This leaves shared memory
// for path states and normals on the target SM89. The launch-time device
// budget check still applies to every geometry and GPU. The heterogeneous
// rough scan without a rate has a global-memory L fallback and is separate.
inline constexpr std::size_t kMaximumSharedCholeskyFactors = 128U;

}  // namespace ai_factory::workbench::monte_carlo
