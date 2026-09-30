// Flat-curve parameter domain shared by host loaders and device preparation.
#pragma once

#include "curve/flat/parameters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::curve::flat {

__host__ __device__ inline bool finite_parameter(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool valid_parameters(
    const FlatCurveParameters& curve
) {
    return finite_parameter(curve.rate);
}

inline void validate_parameters(
    const FlatCurveParameters& curve,
    const std::string& prefix
) {
    if (!std::isfinite(curve.rate))
        throw std::invalid_argument(prefix + "rate must be finite.");
}

}  // namespace ai_factory::workbench::curve::flat
